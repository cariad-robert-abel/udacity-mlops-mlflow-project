#!/usr/bin/env python
"""
Perform basic data cleaning on W&B dataset, upload as new artifact.
"""
import argparse
import logging
import tempfile

import pandas as pd
import wandb

from wandb_utils import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def clean_artifact(df: pd.DataFrame, min_price: int, max_price: int) -> pd.DataFrame:
    """Clean the data and return the cleaned dataframe.

    Args:
        df: The input dataframe to be cleaned.
        min_price: Minimum accepted price.
        max_price: Maximum accepted price.

    Returns:
        The cleaned dataframe.
    """
    logger.info(f'Cleaning {df.shape[0]} rows of data...')

    # Remove outliers based on price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    return df

def go(args: argparse.Namespace):

    run = wandb.init(job_type='clean-data')
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f'Downloading artifact {args.input_artifact}')
    filename = run.use_artifact(args.input_artifact).file()

    with open(filename, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f, low_memory=False)

    result = clean_artifact(df, args.min_price, args.max_price)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f'{tmpdir}/{args.output_artifact}'
        result.to_csv(filename, index=False, encoding='utf-8')

        logger.info(f'Uploading artifact {args.output_artifact}')
        log_artifact(
            args.output_artifact,
            args.output_type,
            args.output_description,
            filename,
            run
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Very basic Data Cleaning')

    parser.add_argument(
        "--input-artifact",
        type=str,
        help='Name of the input CSV artifact',
        required=True
    )
    parser.add_argument(
        "--output-artifact",
        type=str,
        help='Name for the output artifact',
        required=True
    )
    parser.add_argument(
        "--output-type",
        type=str,
        help='Type of the output artifact',
        required=True
    )
    parser.add_argument(
        "--output-description",
        type=str,
        help='Brief description of the output artifact',
        required=True
    )
    parser.add_argument(
        "--min-price",
        type=int,
        help='Minimum accepted price',
        required=True
    )
    parser.add_argument(
        "--max-price",
        type=int,
        help='Maximum accepted price',
        required=True
    )

    args = parser.parse_args()

    go(args)
