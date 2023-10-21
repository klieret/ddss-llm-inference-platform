import argparse

from llm_inference_platform import deploy, hf_model_downloader


def get_cli() -> argparse.ArgumentParser:
    """Get main command line interface"""
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
    """Run main command line interface"""
    parser = get_cli()
    parser.parse_args()


if __name__ == "__main__":
    main()
