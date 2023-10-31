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

from huggingface_hub import snapshot_download

HF_DEFAULT_HOME = os.environ.get("HF_HOME", ".")


def download_save_huggingface_model(
    repo_id: str, revision: str, cache_dir: str
) -> None:
    """Download model from huggingface hub and save to cache_dir"""
    if cache_dir == "":
        cache_dir = HF_DEFAULT_HOME
    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments to existing parser"""
    parser.add_argument("--repo-id", type=str, help="HF Model Hub Repo ID")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=HF_DEFAULT_HOME,
        help="Location to save model, defaults to None",
    )


def download_cli(args: argparse.Namespace) -> None:
    """Run script from command line"""
    download_save_huggingface_model(args.repo_id, args.revision, args.cache_dir)
