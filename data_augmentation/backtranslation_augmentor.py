import argparse
import logging
import os
from datasets import load_dataset
from tqdm import tqdm
import argostranslate.translate


def configure_logging(level: str) -> None:
    """
    Configure logging format and level.

    Args:
        level (str): Logging level string, e.g., 'INFO', 'DEBUG'.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=getattr(logging, level.upper(), logging.INFO)
    )


def augment_text(text: str, src_lang: str = "en", pivot_lang: str = "de") -> str:
    """
    Perform back-translation: src_lang -> pivot_lang -> src_lang.

    Args:
        text (str): Original text to augment.
        src_lang (str): Source language code.
        pivot_lang (str): Pivot language code.

    Returns:
        str: Augmented text. Returns original text on failure.
    """
    try:
        # Translate to pivot language
        intermediate = argostranslate.translate.translate(text, src_lang, pivot_lang)
        # Translate back to source language
        result = argostranslate.translate.translate(intermediate, pivot_lang, src_lang)
        return result
    except Exception as e:
        logging.warning(
            "Back-translation failed for text: %.50s... (%s)",
            text, e
        )
        return text


def augment_batch(batch: dict, src_lang: str, pivot_lang: str) -> dict:
    """
    Apply back-translation to a batch of texts.

    Args:
        batch (dict): Batch containing a list of strings under key 'anchor'.
        src_lang (str): Source language code.
        pivot_lang (str): Pivot language code.

    Returns:
        dict: Mapping containing key 'text_aug' with list of augmented texts.
    """
    texts = batch.get('anchor', [])
    return {'text_aug': [augment_text(txt, src_lang, pivot_lang) for txt in texts]}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Back-translation augmentation using HuggingFace Datasets and Argos Translate"
    )
    parser.add_argument(
        '--input_path', type=str, required=True,
        help='Path to input txt file with one sentence per line.'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Path to output CSV file (must end with .csv).'
    )
    parser.add_argument(
        '--input_format', choices=['txt'], default='txt',
        help='Format of input data.'
    )
    parser.add_argument(
        '--src_lang', type=str, default='en',
        help='Source language code.'
    )
    parser.add_argument(
        '--pivot_lang', type=str, default='de',
        help='Pivot language code for back-translation.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for dataset.map.'
    )
    parser.add_argument(
        '--num_proc', type=int, default=4,
        help='Number of parallel processes.'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        help='Logging level.'
    )
    return parser.parse_args()


def main() -> None:
    """
    Load dataset, perform back-translation in parallel, and save results.
    """
    args = parse_args()
    configure_logging(args.log_level)

    # Load input dataset
    if args.input_format == 'txt':
        logging.info("Loading dataset from %s", args.input_path)
        raw_ds = load_dataset(
            'text',
            data_files={'train': args.input_path}
        )['train']
        dataset = raw_ds.rename_column('text', 'anchor')
    else:
        logging.error("Unsupported input format: %s", args.input_format)
        raise ValueError(f"Unsupported input format: {args.input_format}")

    # Apply back-translation augmentation
    logging.info(
        "Starting augmentation: src=%s, pivot=%s, batch=%d, processes=%d",
        args.src_lang, args.pivot_lang,
        args.batch_size, args.num_proc
    )
    augmented = dataset.map(
        lambda batch: augment_batch(batch, args.src_lang, args.pivot_lang),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc='Augmenting text'
    )

    # Rename columns and save outputs
    result_ds = augmented.rename_column('anchor', 'text')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    logging.info("Saving augmented CSV to %s", args.output_path)
    result_ds.to_csv(args.output_path, index=False)

    hfds_path = args.output_path.replace('.csv', '_hfds')
    logging.info("Saving HuggingFace dataset to %s", hfds_path)
    result_ds.save_to_disk(hfds_path)

    logging.info("Augmentation complete.")


if __name__ == '__main__':
    main()


#python backtranslation_augmentor.py --input_path /home/user/models/data/wikipedia_sentences/wiki1m_for_simcse.txt --output_path /home/user/models/data/wikipedia_sentences/wiki1m_for_simcse.csv --input_format txt --num_proc 8 --batch_size 32 --log_level INFO
