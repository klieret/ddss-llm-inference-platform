#!/usr/bin/env python
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

# fixme
# pylint: disable=missing-function-docstring


def list_available_models(
    model_directory: os.PathLike = Path("./models"),
) -> list[os.PathLike]:
    model_list = sorted(Path(model_directory).iterdir())
    return [m for m in model_list if m.is_dir()]


def construct_cmd(
    model_name: str | os.PathLike = "<SELECT MODEL>",
    model_dir: os.PathLike = Path("./models"),
    quantization: str = "None",
    context_length: int = 2048,
    container_type="docker",  # Docker or Singularity
) -> list[str]:
    # Construct argument string
    cmd = []
    # Containerization arguments
    match container_type:
        case "docker":
            cmd += [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--shm-size",
                "1g",
                "-v",
                f"{model_dir.absolute()}:/data",
                "ghcr.io/huggingface/text-generation-inference:latest",
            ]
        case "singularity":
            cmd += [
                "singularity",
                "run",
                "--nv",
                "--mount",
                f"type=bind,src={model_dir.absolute()},dst=/data",
                "--env",
                "HF_HOME=/data",
                "--env",
                "HF_HUB_OFFLINE=1",
                "text-generation-inference_latest.sif",
            ]
        case _:
            msg = f"{container_type} is not supported."
            raise ValueError(msg)

    # Arguments to text-generation-launcher
    cmd += [
        f"--max-total-tokens={context_length}",
        f"--quantization={quantization}",
        f"--model-name={model_name}",
    ]
    return cmd


def launch_container(cmd: list[str]) -> None:
    # Launch
    logging.info("Running %s", " ".join(cmd))
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logging.info("Launching process with PID %s", process.pid)
