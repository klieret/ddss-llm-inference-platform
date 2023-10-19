#!/usr/bin/env python
from __future__ import annotations

import argparse
import atexit
import os
import shlex
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import jinja2

from llm_inference_platform.slurm import (
    WaitTillRunning,
    cancel_slurm_job,
    get_slurm_node,
)
from llm_inference_platform.ssh import find_open_port, forward_port
from llm_inference_platform.utils.log import logger

# fixme
# pylint: disable=missing-function-docstring


def list_available_models(
    model_directory: str | os.PathLike[Any] = Path("./models"),
) -> list[Path]:
    """Given a model directory, list all model subdirectories"""
    model_list = sorted(Path(model_directory).iterdir())
    return [m for m in model_list if m.is_dir()]


def construct_docker_cmd(
    model_name: str,
    model_dir: Path = Path("./models"),
    quantization: str = "None",
    context_length: int = 2048,
) -> list[str]:
    return [
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
        f"--max-total-tokens={context_length}",
        f"--quantization={quantization}",
        f"--model_name={model_name}",
    ]


def construct_singularity_cmd(
    *,
    model_name: str,
    revision: str | None,
    model_dir: Path = Path("./models"),
    quantization: str | None = None,
    context_length: int = 2048,
    extra_args: list[str] | None = None,
) -> list[str]:
    if extra_args is None:
        extra_args = []
    if revision is None:
        raise NotImplementedError
    model_id = f"{model_name}/snapshots/{revision}"
    cmd = [
        "singularity",
        "run",
        "--nv",
        "--mount",
        f"type=bind,src={model_dir.absolute()},dst=/data",
        "--env",
        "HF_HUB_OFFLINE=1",
        "text-generation-inference_latest.sif",
        f"--huggingface-hub-cache={model_dir.absolute()}",
        f"--max-total-tokens={context_length}",
        f"--model-id={model_id}",
        f"--revision={revision}",
        "--env",
        "--port=8000",
        *extra_args,
    ]
    if quantization is not None:
        cmd.append(f"--quantize={quantization}")
    return cmd


def launch_container(cmd: list[str]) -> None:
    # Launch
    logger.info("Running %s", shlex.join(cmd))
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info("Launching process with PID %s", process.pid)


def format_slurm_submission_script(cmd: list[str], email: str = "") -> str:
    """Format a SLURM submission script

    Args:
        cmd: Command to run
        email: Email to send SLURM notifications to
    """
    assert cmd
    template_file = Path(__file__).parent / "slurm_template.slurm"
    assert template_file.is_file()
    with template_file.open() as f:
        template = jinja2.Template(f.read())
    return template.render(cmd=shlex.join(cmd), email=email)  # type: ignore[no-any-return]


def sbatch(script: str) -> str:
    """Submits a SLURM script to the cluster

    Returns:
        Job ID
    """
    f = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        mode="w", delete=False
    )
    f.write(script)
    f.close()
    logger.debug("Wrote SLURM script to %s", f.name)
    cmd = ["sbatch", f.name]
    logger.debug("Submitting SLURM job: %s", shlex.join(cmd))
    output: str = subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        encoding="UTF-8",
    )
    logger.debug("Output: %s", output)
    prefix = "Submitted batch job "
    if prefix not in output:
        msg = f"Could not submit job: {output}"
        raise RuntimeError(msg)
    job_id = output.split(prefix)[1].strip()
    assert job_id
    return job_id


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a container to the SLURM cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the model to run",
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Revision of the model",
        required=True,
    )
    parser.add_argument(
        "--dir",
        help="Directory containing models",
        default=Path("./models"),
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
        "--extra-args",
        nargs="+",
        help="Extra arguments to pass to text-generation-inference",
        default=None,
    )
    return parser


def main() -> None:
    args = cli().parse_args()
    assert args.dir.is_dir()
    cmd = construct_singularity_cmd(
        model_name=args.name,
        model_dir=args.dir,
        quantization=args.quantization,
        context_length=args.context_length,
        extra_args=args.extra_args,
        revision=args.revision,
    )
    logger.debug("Singularity command: %s", shlex.join(cmd))
    script = format_slurm_submission_script(cmd)
    job_id = sbatch(script)
    # Doesn't matter if slurm job is already dead before we call this
    # cleanup, but need to make sure it never survives
    atexit.register(partial(cancel_slurm_job, job_id))
    wtr = WaitTillRunning(job_id)
    success = wtr.wait()
    if not success:
        msg = "Job failed to start, check SLURM log for details."
        logger.critical(msg)
        sys.exit(234)
    port = find_open_port()
    node = get_slurm_node(job_id)
    logger.info("Forwarding port 8000 on %s to localhost:%s", port, node)
    forward_process = forward_port(node, port, 8000)
    atexit.register(
        lambda: forward_process.terminate()  # pylint: disable=unnecessary-lambda
    )
    input("Press any key to quit.")


if __name__ == "__main__":
    main()
