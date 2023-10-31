from llm_inference_platform.cli import get_cli


def test_cli_deploy() -> None:
    parser = get_cli()
    args = parser.parse_args(["deploy", "--name", "test_model", "--revision", "test"])
    assert args.func.__name__ == "deploy_cli"
    assert args.name == "test_model"


def test_cli_model_dl() -> None:
    parser = get_cli()
    args = parser.parse_args(
        [
            "model-dl",
            "--repo-id",
            "test_model",
            "--revision",
            "test",
            "--cache-dir",
            "test",
        ]
    )
    assert args.func.__name__ == "download_cli"
    assert args.repo_id == "test_model"
    assert args.cache_dir == "test"
