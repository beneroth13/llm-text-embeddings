# this code is based on the following file: https://github.com/ZBWpro/PretCoTandKE/blob/main/train.py
import os
import random
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader as _TorchDataLoader
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, set_seed, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import csv

# ─── Monkey-patch DataLoader to drop "in_order" kwarg ───
_orig_init = _TorchDataLoader.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("in_order", None)
    _orig_init(self, *args, **kwargs)
_TorchDataLoader.__init__ = _patched_init
# ────────────────────────────────────────────────────────

# Load configuration
import settings

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
)

class PairedCollator:
    def __init__(self, tokenizer, template: str):
        self.tokenizer = tokenizer
        self.template  = template
    def _apply_noise(self, text: str) -> str:
        if settings.DO_DELETION:
            tokens = text.split()
            if len(tokens) > 1:
                keep = [tok for tok in tokens if random.random() > 0.1]
                text = " ".join(keep) if keep else random.choice(tokens)
        if settings.DO_SWAP:
            tokens = text.split()
            if len(tokens) > 1:
                i, j = random.sample(range(len(tokens)), 2)
                tokens[i], tokens[j] = tokens[j], tokens[i]
                text = " ".join(tokens)
        if settings.DO_CHAR_NOISE:
            tokens = text.split()
            for idx, tok in enumerate(tokens):
                if random.random() < 0.05 and len(tok) > 1:
                    tl = list(tok)
                    k = random.randint(0, len(tl)-2)
                    tl[k], tl[k+1] = tl[k+1], tl[k]
                    tokens[idx] = ''.join(tl)
            text = " ".join(tokens)
        return text
    def __call__(self, features):
        origs, augs = [], []
        for ex in features:
            o = self.tokenizer.decode(ex['input_ids'], skip_special_tokens=True)
            u = self.tokenizer.decode(ex['input_ids_aug'], skip_special_tokens=True)
            o = self._apply_noise(o)
            u = self._apply_noise(u)
            origs.append(self.template.replace("*sent_0*", o))
            augs.append(self.template.replace("*sent_0*", u))
        tok_o = self.tokenizer(origs, padding=False, truncation=False, add_special_tokens=False)
        tok_u = self.tokenizer(augs, padding=False, truncation=False, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id if settings.ADD_EOS else None
        def build_list(ids_list, msk_list):
            res_ids, res_msk = [], []
            for ids, msk in zip(ids_list, msk_list):
                if eos is not None:
                    ids = ids + [eos]
                    msk = msk + [1]
                res_ids.append(torch.tensor(ids, dtype=torch.long))
                res_msk.append(torch.tensor(msk, dtype=torch.long))
            return res_ids, res_msk
        o_ids, o_msk = build_list(tok_o['input_ids'], tok_o['attention_mask'])
        u_ids, u_msk = build_list(tok_u['input_ids'], tok_u['attention_mask'])
        max_len = max(t.shape[0] for t in o_ids + u_ids)
        def pad(t):
            return t if t.shape[0] == max_len else F.pad(t, (max_len - t.shape[0], 0), value=0)
        return {
            'input_ids':        torch.stack([pad(x) for x in o_ids]),
            'attention_mask':   torch.stack([pad(x) for x in o_msk]),
            'input_ids_aug':    torch.stack([pad(x) for x in u_ids]),
            'attention_mask_aug':torch.stack([pad(x) for x in u_msk])
        }

class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SentembTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_mode = settings.POOL_MODE
        self.sim = None
    # Accept extra kwargs like num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ids = torch.cat([inputs['input_ids'], inputs['input_ids_aug']], dim=0)
        msk = torch.cat([inputs['attention_mask'], inputs['attention_mask_aug']], dim=0)
        out = model(input_ids=ids, attention_mask=msk, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        last = h[:, -1, :]
        mask = msk.unsqueeze(-1).type_as(h)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pool_mode == 'last': pool = last
        elif self.pool_mode == 'mean': pool = mean
        else: pool = torch.cat([last, mean], dim=-1)
        B2, D = pool.shape
        B = B2 // 2
        z1, z2 = pool[:B], pool[B:]
        if dist.is_initialized():
            ws = dist.get_world_size()
            z1s = [torch.zeros_like(z1) for _ in range(ws)]
            z2s = [torch.zeros_like(z2) for _ in range(ws)]
            dist.all_gather(z1s, z1.contiguous())
            dist.all_gather(z2s, z2.contiguous())
            z1s[dist.get_rank()] = z1
            z2s[dist.get_rank()] = z2
            z1 = torch.cat(z1s, dim=0)
            z2 = torch.cat(z2s, dim=0)
        if self.sim is None:
            self.sim = Similarity(settings.TEMPERATURE).to(z1.device)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        N = cos_sim.size(0)
        mask_diag = torch.eye(N, device=cos_sim.device, dtype=torch.bool)
        logits = cos_sim.clone()
        logits[mask_diag] -= settings.MARGIN
        labels = torch.arange(N, device=cos_sim.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, (z1, z2)) if return_outputs else loss


def process_example(ex, tokenizer, cutoff_len):
    tok_o = tokenizer(ex['text'], truncation=True, max_length=cutoff_len, add_special_tokens=False)
    tok_u = tokenizer(ex['text_aug'], truncation=True, max_length=cutoff_len, add_special_tokens=False)
    return {
        'input_ids': tok_o['input_ids'],
        'attention_mask': tok_o['attention_mask'],
        'input_ids_aug': tok_u['input_ids'],
        'attention_mask_aug': tok_u['attention_mask']
    }


def main():
    set_seed(settings.SEED)
    wandb.init(project=settings.WANDB_PROJECT)

    cfg = AutoConfig.from_pretrained(settings.BASE_MODEL, cache_dir=settings.CACHE_DIR)
    cfg.hidden_dropout_prob = settings.DROPOUT
    cfg.attention_dropout = settings.DROPOUT
    cfg.output_hidden_states = True

    # 4-bit model loading without redundant load_in_4bit arg
    if settings.LOAD_KBIT == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            bnb_4bit_compute_dtype=torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL,
            config=cfg,
            quantization_config=quant_cfg,
            device_map='auto',
            cache_dir=settings.CACHE_DIR
        )
        peft_cfg = LoraConfig(
            r=settings.LORA_R,
            lora_alpha=settings.LORA_ALPHA,
            target_modules=settings.LORA_TARGET_MODULES,
            lora_dropout=settings.LORA_DROPOUT,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, peft_cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL,
            config=cfg,
            load_in_8bit=(settings.LOAD_KBIT == 8),
            torch_dtype=torch.float16 if settings.LOAD_KBIT == 16 else torch.float32,
            device_map='auto',
            cache_dir=settings.CACHE_DIR
        )
        peft_cfg = LoraConfig(
            r=settings.LORA_R,
            lora_alpha=settings.LORA_ALPHA,
            target_modules=settings.LORA_TARGET_MODULES,
            lora_dropout=settings.LORA_DROPOUT,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, peft_cfg)

    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL, cache_dir=settings.CACHE_DIR)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    raw = load_dataset(
        'csv', data_files={'train': settings.DATA_PATH},
        column_names=['text','text_aug'], header=None
    )
    raw['train'] = raw['train'].filter(lambda x: x['text'].strip() and x['text_aug'].strip())
    data = raw['train']
    train_ds = data.map(
        process_example,
        fn_kwargs={'tokenizer': tokenizer, 'cutoff_len': settings.CUTOFF_LEN},
        remove_columns=['text','text_aug'],
        desc='Preprocessing',
        num_proc=settings.NUM_PROC
    )

    trainer = SentembTrainer(
    model=model,
    train_dataset=train_ds,
    args=TrainingArguments(
        per_device_train_batch_size=settings.MICRO_BATCH_SIZE,
        gradient_accumulation_steps=settings.BATCH_SIZE // settings.MICRO_BATCH_SIZE,
        warmup_steps=settings.WARMUP_STEPS,
        num_train_epochs=settings.NUM_EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        fp16=settings.FP16,
        logging_steps=settings.LOG_STEPS,
        save_steps=settings.SAVE_STEPS,
        output_dir=settings.OUTPUT_DIR,
        report_to='wandb',
        group_by_length=settings.GROUP_BY_LENGTH,
        remove_unused_columns=False  # keep augmented fields for collator
    ),
    data_collator=PairedCollator(tokenizer, settings.PROMPT_TEMPLATE)
)

    if torch.__version__ >= '2' and os.name != 'nt':
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(settings.OUTPUT_DIR)

if __name__ == '__main__':
    main()
