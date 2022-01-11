#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import os

import pandas as pd
from wandb.apis.public import File


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading and reading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    df = pd.read_csv(artifact_local_path, low_memory=False)

    # Drop outliers
    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # test_proper_boundaries check
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    filename = args.output_artifact
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument("--input_artifact", type=str, help="Name of the input artifact", required=True)

    parser.add_argument("--output_artifact", type=str, help="Name of the output artifact", required=True)

    parser.add_argument("--output_type", type=str, help="Output artifact type", required=True)

    parser.add_argument("--output_description", type=str, help="A brief description of this artifact", required=True)

    parser.add_argument("--min_price", type=float, help="Min price threshold to drop outliers", required=True)

    parser.add_argument("--max_price", type=float, help="Max price threshold to drop outliers", required=True)

    args = parser.parse_args()

    go(args)
