import os
import re
import random
import argparse
import logging
import csv
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def contains_chinese(text: str) -> bool:
    """
    Check if the input text contains any Chinese characters.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if Chinese characters are found, False otherwise.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def generate_augmented_batch(
    sentences: list[str],
    prompt_templates: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 100,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0
) -> tuple[list[str], list[str]]:
    """
    Generate augmented versions of input sentences using a language model.

    Args:
        sentences (list[str]): List of original sentences.
        prompt_templates (list[str]): List of templates for prompts.
        tokenizer: Pretrained Transformers tokenizer.
        model: Pretrained Transformers causal LM model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p (nucleus) sampling probability.
        repetition_penalty (float): Penalty for repeating tokens.

    Returns:
        tuple: (generated_texts, prompts_used)
    """
    # Build prompts
    prompts = [
        random.choice(prompt_templates).format(sentence=sent) for sent in sentences
    ]

    # Chat messages
    messages_batch = [
        [
            {"role": "system", "content": (
                "You are a helpful assistant. Always respond only in English."
                " Do not use any Chinese or other languages."
            )},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    # Tokenize and move to device
    inputs = (
        tokenizer.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        .to(model.device)
    )

    # Generate outputs
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode new tokens only
    seq_start = inputs.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, seq_start:],
        skip_special_tokens=True
    )

    return outputs, prompts


def configure_logging(level: str) -> None:
    """
    Configure logging format and level.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=getattr(logging, level.upper(), logging.INFO)
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate augmented sentence pairs using a pretrained chat model"
    )
    parser.add_argument(
        "--model-id", type=str,
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="Pretrained model identifier"
    )
    parser.add_argument(
        "--device-id", type=str,
        default="0",
        help="CUDA device ID"
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="Path to input CSV with a 'text' column"
    )
    parser.add_argument(
        "--output-csv", type=str, required=True,
        help="Path to save augmented outputs"
    )
    parser.add_argument(
        "--skipped-csv", type=str, required=True,
        help="Path to log skipped batches"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of sentences per batch"
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Maximum retries for batches with invalid output"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: load model, process input CSV, and write augmentations.
    """
    args = parse_args()
    configure_logging(args.log_level)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    logging.info("Loading model '%s' on device %s", args.model_id, args.device_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side='left'
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Default prompts
    prompt_templates = [
        "Rewrite the following sentence with more detail, keeping the original meaning. Respond with only the rewritten sentence:\n\n'{sentence}'",
        "Paraphrase this sentence using simpler language. Respond with only the rewritten sentence:\n\n'{sentence}'",
        "You're a high school teacher. Explain this sentence to your students in your own words. Respond with only the explanation:\n\n'{sentence}'",
        "Write a metaphor that expresses the same idea as this sentence. Respond with only the metaphor:\n\n'{sentence}'",
        "After reading this sentence, what is a natural question someone might ask? Respond with only the question:\n\n'{sentence}'"
    ]

    # Read input
    df = pd.read_csv(args.input_csv)
    sentences = df['text'].tolist()

    file_exists = os.path.exists(args.output_csv)
    skipped_exists = os.path.exists(args.skipped_csv)

    # Batch loop
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="Generating augmentations"):
        batch = sentences[i:i + args.batch_size]
        success = False
        retries = 0

        while not success and retries < args.max_retries:
            outputs, prompts = generate_augmented_batch(
                batch, prompt_templates, tokenizer, model
            )
            cleaned = []

            for orig, aug in zip(batch, outputs):
                if contains_chinese(aug):
                    logging.warning(
                        "Chinese detected in batch %d (retry %d/%d)",
                        i, retries+1, args.max_retries
                    )
                    break
                cleaned.append({'text': orig, 'text_aug': aug.strip()})
            else:
                pd.DataFrame(cleaned).to_csv(
                    args.output_csv,
                    mode='a',
                    header=not file_exists,
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC,
                    lineterminator="\n",
                    escapechar="\\"
                )
                file_exists = True
                success = True

            retries += 1

        if not success:
            logging.error(
                "Skipping batch %d after %d failed attempts", i, args.max_retries
            )
            pd.DataFrame({'text': batch}).to_csv(
                args.skipped_csv,
                mode='a',
                header=not skipped_exists,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                lineterminator="\n",
                escapechar="\\"
            )
            skipped_exists = True

    logging.info("Done. Augmented outputs saved to: %s", args.output_csv)
    logging.info("Skipped batches (if any) logged to: %s", args.skipped_csv)


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=4 python augmented_batch_generator.py --model-id Qwen/Qwen1.5-0.5B-Chat --device-id 4 --input-csv /home/user/models/data/augmentation_inputs/toxic_conversations_50k_text.csv --output-csv /home/user/models/data/augmentation_inputs/toxic_conversations_50k_qwen_positives_test.csv --skipped-csv /home/user/models/data/augmentation_inputs/toxic_conversations_50k_skipped_batches_test.csv --batch-size 8 --max-retries 5 --log-level INFO
