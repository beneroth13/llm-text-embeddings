# llm-text-embeddings

> LoRA fine-tuning & MTEB benchmarking toolkit  
> Build text embeddings from causal LMs (e.g. Qwen, Llama), with optional prompt-wrapping and PEFT adapters.

---

## 📂 Repository structure
```
llm-text-embeddings/
├── benchmark/
│   ├── benchmark.py
│   └── collect_scores.py
├── data_augmentation/
│   ├── augmented_batch_generator.py
│   └── backtranslation_augmentor.py
├── finetuning/
│   ├── train.py
│   └── settings.py
├── requirements/
│   ├── environment-finetuning.yml
│   └── environment-mteb-benchmark.yml
├── attention.ipynb
└── README.md
```

- **benchmark/**  
  Scripts for running MTEB evaluations.
- **data_augmentation/**  
  Creation of positive pairs through either translation or prompting Qwen.
- **finetuning/**  
  Scripts & settings for LoRA-style contrastive fine-tuning.

---

## ⚙️ Installation

1. **Clone** the repo:
   ```bash
   git clone https://github.com/beneroth13/llm-text-embeddings.git
   cd llm-text-embeddings
   ```

2. Create & activate Python environments (recommended):
   ```bash
   # create the environment
    conda env create -n mteb-benchmark -f requirements/environment-mteb-benchmark.yml

    # activate it
    conda activate mteb-benchmark
   ```
   or
   ```bash
   # create the environment
    conda env create -n finetuning -f requirements/environment-finetuning.yml

    # activate it
    conda activate finetuning  
   ```

   mteb-benchmark is used for the files in benchmark, finetuning for all the remaining files.<br><br><br>

## Dataset augmentations
You can use any arbitrary text dataset for the augmentations. In the following, you can see examples for the application of the code:
```bash
CUDA_VISIBLE_DEVICES=2 \
  python augmented_batch_generator.py \
  --model-id Qwen/Qwen1.5-0.5B-Chat \
  --device-id 2 \
  --input-csv "/home/user/models/data/augmentation_inputs/toxic_conversations_50k_text.csv" \
  --output-csv "/home/user/models/data/augmentation_inputs/test/toxic_conversations_50k_qwen_positives.csv" \
  --skipped-csv "/home/user/models/data/augmentation_inputs/test/toxic_conversations_50k_skipped_batches.csv" \
  --batch-size 16 \
  --max-retries 5 \
  --log-level INFO
```

```bash
python backtranslation_augmentor.py \
  --input_path "/home/user/models/data/wikipedia_sentences/small_test.txt" \
  --output_path "/home/user/models/data/wikipedia_sentences/small_test.csv" \
  --input_format txt \
  --num_proc 8 \
  --batch_size 32 \
  --log_level INFO
```

We have created our own datasets with positive pairs. These, as well as other datasets we used, can be found in the section Data.

## 🛠️ Fine-tuning (LoRA)
Use your paired CSV (text,text_aug) to train a LoRA adapter, all the details are set in settings.py:
```bash
python train.py 
```
In setting.py, all the hyperparameters can be specified, as well as which augmentations to use, which prompt, and which pooling method.
The code is based on this file from [PretCoTandKE](https://github.com/ZBWpro/PretCoTandKE/blob/main/train.py).

## 📊 Benchmarking (MTEB)
Encode with your model (base or LoRA-tuned) and evaluate on MTEB tasks:
```bash
python benchmark.py \
  --save-path "/home/user/code/output/Benchmarking/Test/LLama_lora_classification_last_withEOS" \
  --model-name llama \
  --aggregation last \
  --batch-size 16 \
  --task-types Classification \
  --lora-checkpoint "/home/user/data/LLM2vec/llama_lora/llama-lora-PromptClustering_last_dropout0.05_bs120_lre-5_temp0.2_qwenwiki_noEOS_6layers/checkpoint-100" \
  --prompt "classification" \
  --append-eos

```
The benchmarking is based on the code and package from [MTEB](https://github.com/embeddings-benchmark/mteb/blob/main/README.md).

## 📥 Data

We use the following public datasets as sources for augmentation:

- **[Wikipedia Sentences (wiki1m_for_simcse)](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse)**  
  1 million English Wikipedia sentences for contrastive learning.
- **[Toxic Conversations (50k chat logs)](https://huggingface.co/datasets/mteb/toxic_conversations_50k)**  
  50k conversational snippets labelled for toxicity.
- **[Emotion Tweets (6-class labeled Twitter data)](https://huggingface.co/datasets/dair-ai/emotion)**  
  Tweets annotated with one of six emotions: anger, fear, joy, love, sadness, surprise.

This [NLI](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv) dataset from SimCSE already has positive pairs and was used as well.

### Downloaded / Augmented datasets

We also provide our own positive-pair CSVs (`text,text_aug`):

| Name                                   | Description                        | Download                                                                              |
|----------------------------------------|------------------------------------|---------------------------------------------------------------------------------------|
| `wiki1m_for_simcse_positive_pairs_qwen_prompted.csv` | Qwen-generated paraphrases          | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/wiki1m_for_simcse_positive_pairs_qwen_prompted.csv)                      |
| `wiki1m_for_simcse_positive_pairs_translate.csv` | Back-translated Wikipedia samples           | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/wiki1m_for_simcse_positive_pairs_translate.csv)                      |
| `emotion_positive_pairs_qwen_prompted.csv`   | Qwen-generated emotion paraphrases   | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/emotion_positive_pairs_qwen_prompted.csv)                              |
| `toxic_conversations_50k_positive_pairs_qwen_prompted.csv`           | Qwen-generated toxic conversation paraphrases  | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/toxic_conversations_50k_positive_pairs_qwen_prompted.csv)                                      |

*(Datasets are provided in the GitHub “Releases” section.)*

---

## 🚀 Checkpoints

Our best LoRA adapters for downstream tasks:

| Model                    | Task                                    | Checkpoint path                                                                                         |
|--------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------|
| Llama-3.2-1B + LoRA + last | Clustering (last-token)             | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/LLama_last_best.zip)                                         |
| Llama-3.2-1B + LoRA + mean        | Clustering (mean pooling)               | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/LLama_mean_best.zip)                                                    |
| Llama-3.2-1B + LoRA + last + emotion       | Emotion Classification   | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/LLama_emotion_last_best.zip)                                                   |
| Llama-3.2-1B + LoRA + last + toxic      | Toxic Conversation Classification  | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/LLama_toxic_last_best.zip)                                                   |
| Qwen3-0.6B + LoRA + last       | Clustering (last-token)   | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/Qwen3_last_best.zip)                                                   |
| Qwen3-0.6B + LoRA + mean       | Clustering (mean pooling)   | [Download](https://github.com/beneroth13/llm-text-embeddings/releases/download/v1.0/Qwen3_mean_best.zip)                                                   |

*(Checkpoints are uploaded under GitHub “Releases”.)*

Please keep in mind that you will have to unzip the checkpoints. Also, the files only contain the adapter, which you can use as input for benchmark.py.

## Citation
TODO after paper is uploaded

## 📜 License
This project is licensed under the MIT License. See LICENSE for details.
