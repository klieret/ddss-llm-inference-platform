import argparse
from pathlib import Path

from llm_inference_platform.deploy import SHARED_RESOURCE_DIR, deploy, get_weight_dir
from llm_inference_platform.hf_model_downloader import (
    HF_DEFAULT_HOME,
    download_save_huggingface_model,
)
from llm_inference_platform.utils.log import logger


def _add_deploy_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for deploy to existing parser"""
    parser.add_argument(
        "--name",
        type=str,
        help=(
            "Name of the model to run. Pass as HuggingFace org_name/model_name, e.g.,"
            "'meta-llama/Llama-2-7b-chat-hf'."
        ),
        default="",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision of the model. Defaults to 'main'.",
    )
    parser.add_argument(
        "--weight-dir",
        type=str,
        help=(
            "Path to the weight directory of the model, e.g., "
            "/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235"
        ),
        default="",
    )
    parser.add_argument(
        "--model-dir",
        help="Directory containing models",
        default=Path(SHARED_RESOURCE_DIR / "models"),
        type=Path,
    )
    parser.add_argument(
        "--quantization",
        type=str,
        help="Quantization method to use",
        default=None,
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context length to use",
        default=2048,
    )
    parser.add_argument(
        "--singularity-image",
        type=Path,
        help=(
            "Path to singularity container. Defaults to "
            "shared image, if changing you must provide "
            "a precompiled singularity image."
        ),
        default=SHARED_RESOURCE_DIR
        / "singularity/text-generation-inference_latest.sif",
    )
    parser.add_argument(
        "--extra-args",
        nargs="+",
        help="Extra arguments to pass to text-generation-inference",
        default=None,
    )


def _deploy_cli(args: argparse.Namespace) -> None:
    """Run deployment from CLI"""
    weight_dir = args.weight_dir
    if not weight_dir:
        if not args.name:
            msg = "Must provide either --name or --weight-dir"
            raise ValueError(msg)
        weight_dir = get_weight_dir(
            model_ref=args.name,
            revision=args.revision,
            model_dir=args.model_dir,
        )
        logger.debug("Using weight directory: %s", weight_dir)
    deploy(
        weight_dir=weight_dir,
        quantization=args.quantization,
        context_length=args.context_length,
        singularity_image=args.singularity_image,
        extra_args=args.extra_args,
    )


def _add_downloader_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for model downloader to existing parser"""
    parser.add_argument("--repo-id", type=str, help="HF Model Hub Repo ID")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=HF_DEFAULT_HOME,
        help="Location to save model, defaults to None",
    )


def _download_cli(args: argparse.Namespace) -> None:
    """Run model downloader from command line"""
    download_save_huggingface_model(args.repo_id, args.revision, args.cache_dir)


def get_cli() -> argparse.ArgumentParser:
    """Get main command line interface"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    deploy_parser = subparsers.add_parser("deploy")
    _add_deploy_cli_args(deploy_parser)
    deploy_parser.set_defaults(func=_deploy_cli)
    model_dl_parser = subparsers.add_parser("model-dl")
    _add_downloader_cli_args(model_dl_parser)
    model_dl_parser.set_defaults(func=_download_cli)
    return parser


def main() -> None:
    """Run main command line interface"""
    parser = get_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
