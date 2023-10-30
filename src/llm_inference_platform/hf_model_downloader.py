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

HF_DEFAULT_HOME = os.environ.get("HF_HOME", ".")


def download_save_huggingface_model(
    repo_id: str, revision: str, cache_dir: str
) -> None:
    """Download model from huggingface hub and save to cache_dir"""
    if cache_dir == "":
        cache_dir = HF_DEFAULT_HOME
    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


def get_weight_dir(
    model_ref: str,
    model_dir: str | os.PathLike[Any] = HF_DEFAULT_HOME,
    revision: str | None = "main",
    snapshot: str | None = None,
) -> str:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        snapshot (str): snapshot hash of model. Defaults to None. If provided, overrides revision.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    model_path = "--".join(["models", *model_ref.split("/")])
    if snapshot is None:
        snapshot = (model_dir / f"{model_path}/refs/{revision}").read_text()
    return f"{model_path}/snapshots/{snapshot}"


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments to existing parser"""
    parser.add_argument("--repo_id", type=str, help="HF Model Hub Repo ID")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=HF_DEFAULT_HOME,
        help="Location to save model, defaults to None",
    )


def download_cli(args: argparse.Namespace) -> None:
    """Run script from command line"""
    download_save_huggingface_model(args.repo_id, args.revision, args.cache_dir)
