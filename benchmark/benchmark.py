import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mteb import MTEB
import mteb
from peft import PeftModel
from mteb.encoder_interface import PromptType
from torch.utils.data import DataLoader as _TorchDataLoader
import logging
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROMPTS = {
    "none":        "{sent}",
    "eol":         'This sentence: "{sent}" means in one word:',
    "cluster1":    'This sentence: "{sent}" belongs to the following cluster:',
    "cluster2":    'Cluster the text: {sent}',
    "sum":         'This sentence: "{sent}" can be summarized as',
    "pcot":        'After thinking step by step, this sentence: "{sent}" means in one word',
    "classification":     'This sentence : "{sent}" can be classified as:',
    "question":    'Which cluster would you assign the sentence: "{sent}" to?',
    "spec_emotion":(
        'Classify the emotion expressed in the given Twitter "{sent}" '
        'into one of the six emotions: anger, fear, joy, love, sadness, and surprise'
    )
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MTEB benchmarking with Qwen or Llama embeddings"
    )
    parser.add_argument(
        "--save-path", required=True,
        help="Directory to write MTEB results"
    )
    parser.add_argument(
        "--model-name", required=True,
        choices=["qwen", "llama"],
        help="Which model to load"
    )
    parser.add_argument(
        "--aggregation", default="mean",
        choices=["mean", "last"],
        help="Pooling strategy"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size"
    )
    parser.add_argument(
        "--task-types", nargs='+', default=["Clustering"],
        help="MTEB task types to include (e.g. Classification, Clustering)"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="Path to a PEFT/LoRA checkpoint to load (if you’ve fine-tuned a model)"
    )
    parser.add_argument(
        "--prompt",
        choices=[
            "none",         # raw sentence
            "eol",          # “This sentence: "{sent}" means in one word:”
            "cluster1",     # “This sentence: "{sent}" belongs to the following cluster:”
            "cluster2",     # “Cluster the text: {sent}”
            "sum",          # “This sentence: "{sent}" can be summarized as”
            "pcot",         # “After thinking step by step…”
            "classification",      # “This sentence : "{sent}" can be classified as:”
            "question",     # “Which cluster would you assign the sentence: "{sent}" to?”
            "spec_emotion"  # Twitter emotion prompt
        ],
        default="none",
        help="Which prompt template to wrap around each sentence"
    )
    parser.add_argument(
        "--append-eos", action="store_true",
        help="Append the EOS token to the input after adding the prompt"
    )
    return parser.parse_args()


class ModelEmbedder:
    """
    Wraps a causal LM for MTEB:
      - Aggregation: mean or last-token
    """
    def __init__(self, model, tokenizer, device, aggregation, batch_size, prompt_key, append_eos):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.prompt_tpl = PROMPTS[prompt_key]
        self.append_eos = append_eos

    def encode(self, sentences, task_name=None, prompt_type: PromptType = None, **kwargs):
        """
        Encode a list of sentences with:
          - truncation BEFORE prompt
          - prompt insertion
          - tokenization AFTER prompt
        """
        MAX_LENGTH = 512
        WRAP_BUFFER = 50  # extra tokens allowed after wrapping
        all_embeddings = []

        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Encoding", unit="batch"):
            batch_sents = sentences[i:i + self.batch_size]

            # Step 1: Pre-truncate raw sentences (no special tokens, no padding)
            pre_truncated = []
            for sent in batch_sents:
                tok = self.tokenizer(
                    sent,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    add_special_tokens=False
                )
                decoded = self.tokenizer.decode(tok["input_ids"], skip_special_tokens=True)
                pre_truncated.append(decoded)

            # Step 2: Insert into prompt template + optional EOS
            wrapped_batch = []
            for trunc_sent in pre_truncated:
                wrapped = self.prompt_tpl.format(sent=trunc_sent)
                if self.append_eos:
                    wrapped += f" {self.tokenizer.eos_token}"
                wrapped_batch.append(wrapped)

            # Step 3: Tokenize wrapped text with truncation & padding
            inputs = self.tokenizer(
                wrapped_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH + WRAP_BUFFER
            ).to(self.device)

            # Step 4: Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1]  # [B, L, D]

            # Step 5: Pooling
            if self.aggregation == 'mean':
                mask = inputs['attention_mask'].unsqueeze(-1)          # [B, L, 1]
                summed = (hidden * mask).sum(dim=1)                    # [B, D]
                lengths = mask.sum(dim=1).clamp(min=1e-9)              # [B, 1]
                embs = (summed / lengths).cpu().numpy()                # [B, D]
            elif self.aggregation == 'last':
                lengths = inputs['attention_mask'].sum(dim=1) - 1      # [B]
                idx = torch.arange(hidden.size(0), device=self.device)
                embs = hidden[idx, lengths, :].cpu().numpy()           # [B, D]
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

            all_embeddings.append(embs)

        return np.vstack(all_embeddings)


def main():
    torch.cuda.empty_cache()
    args = parse_args()

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model_name == 'qwen':
        model_id = 'Qwen/Qwen3-0.6B'
        trust = True
    elif args.model_name == 'llama':
        model_id = 'meta-llama/Llama-3.2-1B'
        trust = False
    else:
        raise ValueError(f"Unknown model '{args.model_name}'")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/user/models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
        cache_dir="/home/user/models"
    )
    if args.lora_checkpoint:
        model = PeftModel.from_pretrained(model, args.lora_checkpoint, is_trainable=False, device_map="auto")
        model.eval()
    model.config.output_hidden_states = True

    # load tasks
    benchmark = mteb.get_tasks(task_types=args.task_types, languages=["eng"])
    skip_tasks = {"YahooAnswersTopicsClassification"}
    #keep_tasks = {"EmotionClassification"} #"EmotionClassification", "ToxicConversationsClassification
    filtered = []
    for task in benchmark:
        name = getattr(task, 'name', type(task).__name__)
        if name in skip_tasks or 'Legal' in name:
            continue
        #elif name not in keep_tasks:
        #    continue
        filtered.append(task)
    if not filtered:
        print("No tasks to run after filtering.")
        return

    # embedder
    embedder = ModelEmbedder(
        model, tokenizer, device,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        prompt_key=args.prompt,
        append_eos=args.append_eos
    )

    # run MTEB
    evaluator = MTEB(tasks=filtered)
    results = evaluator.run(embedder, output_folder=args.save_path)
    print(results)


if __name__ == '__main__':
    main()
