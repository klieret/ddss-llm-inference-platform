#! /usr/bin/python3

# HuggingFace Model Downloader Script
# Author: Dr Musashi Hinck
#
# Convenience command-line script to download model assets to cache
# for offline usage on compute nodes.

# Requirements:
# huggingface_hub

# Usage:
# python hf_model_downloader.py --repo_id='HF_MODEL_REF' --revision='main' --cache_dir=''

# Script begins


import argparse
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "")


def download_save_huggingface_model(
    repo_id: str, revision: str, cache_dir: str
) -> None:
    """Download model from huggingface hub and save to cache_dir"""
    if cache_dir == "":
        cache_dir = HF_DEFAULT_HOME
    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


def get_weight_dir(
    model_ref: str,
    hf_cache_dir: str | os.PathLike[Any] = HF_DEFAULT_HOME,
    revision: str = "main",
) -> Path:
    """
    Convenience function for retrieving locally stored HF weights.
    """
    hf_cache_dir = Path(hf_cache_dir)
    model_path = "--".join(["models", *model_ref.split("/")])
    snapshot = (hf_cache_dir / f"{model_path}/refs/{revision}").read_text()
    return hf_cache_dir / f"{model_path}/snapshots/{snapshot}"


def main() -> None:
    """Run script from command line"""
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, help="HF Model Hub Repo ID")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Location to save model, defaults to None",
    )
    args = parser.parse_args()

    # Call
    download_save_huggingface_model(args.repo_id, args.revision, args.cache_dir)


if __name__ == "__main__":
    main()
