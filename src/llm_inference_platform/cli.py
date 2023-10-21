import argparse

import llm_inference_platform.deploy as deploy
import llm_inference_platform.hf_model_downloader as hf_model_downloader


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    deploy_parser = subparsers.add_parser("deploy")
    deploy.add_cli_options(deploy_parser)
    deploy_parser.set_defaults(func=deploy.deploy_cli)
    model_dl_parser = subparsers.add_parser("model-dl")
    hf_model_downloader.add_cli_args(model_dl_parser)
    model_dl_parser.set_defaults(func=hf_model_downloader.download_cli)
    return parser


def main() -> None:
    parser = get_cli()
    parser.parse_args()


if __name__ == "__main__":
    main()
