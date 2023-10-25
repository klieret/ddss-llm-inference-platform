from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import jinja2

from llm_inference_platform.slurm import (
    JobState,
    WaitTillRunning,
    cancel_slurm_job,
    get_slurm_job_status,
    get_slurm_node,
    sbatch,
)
from llm_inference_platform.ssh import find_open_port, forward_port
from llm_inference_platform.utils.log import DEFAULT_LOGGER_PATH, logger


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
    """Run text-generation-inference in a Docker container

    See `construct_singularity_cmd` for parameters.
    """
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
    """Run ``text-generation-inference`` in singularity container

    Args:
        model_name (str): Name of model
        revision (str | None): Version/revision of model
        model_dir (Path, optional): Directory with models saved for offline use.
            Defaults to Path("./models").
        quantization (str | None, optional): Quantization of model. Defaults to None.
        context_length (int, optional): Context length of model. Defaults to 2048.
        extra_args (list[str] | None, optional): Extra arguments passed to
            ``text-generation-inference``. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        list[str]: _description_
    """
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


def add_cli_options(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments to existing parser"""
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


class PersistInfo(NamedTuple):
    """Information to write to a file in user's home directory"""

    job_id: str
    port: str
    node: str

    def dump(self, path: Path) -> None:
        """Dump information to a file"""
        with path.open("w") as f:
            json.dump(self._asdict(), f)  # pylint: disable=no-member

    @classmethod
    def from_file(cls, path: Path) -> PersistInfo:
        """Load information from a file"""
        with path.open("r") as f:
            dct = json.load(f)
        return cls(**dct)


def terminate_process(process: subprocess.Popen[Any]) -> None:
    """Terminate process and log it"""
    logger.debug("Terminating process with PID %s", process.pid)
    process.terminate()


def print_debug_information(job_id: str | None = None) -> None:
    """Print debug information at the end of the script"""
    print(  # noqa: T201
        "If this script failed or did not work as expected, please include the "
        "debug output in your report. It is saved in the file: "
        f"{DEFAULT_LOGGER_PATH}. "
    )
    if job_id is not None:
        log_file = Path(f"llm-inference-platform-{job_id}.log")
        if log_file.is_file():
            print(  # noqa: T201
                f"Please also include the SLURM log file for job: {log_file}"
            )


def deploy(**kwargs) -> None:  # type: ignore[no-untyped-def]
    """Deploy a model to the cluster.

    Args:
        See `construct_singularity_cmd` for arguments.
    """
    atexit.register(print_debug_information)
    cmd = construct_singularity_cmd(**kwargs)
    logger.debug("Singularity command: %s", shlex.join(cmd))
    script = format_slurm_submission_script(cmd)
    job_id = sbatch(script)
    # Doesn't matter if slurm job is already dead before we call this
    # cleanup, but need to make sure it never survives
    atexit.register(cancel_slurm_job, job_id)
    wtr = WaitTillRunning(job_id)
    atexit.unregister(print_debug_information)
    atexit.register(print_debug_information, job_id)
    success = wtr.wait()
    if not success:
        msg = "Job failed to start, check SLURM log for details."
        logger.critical(msg)
        sys.exit(234)
    port = find_open_port()
    node = get_slurm_node(job_id)
    logger.info("Forwarding port 8000 on %s to localhost:%s", port, node)
    forward_process = forward_port(node, port, 8000)
    atexit.register(terminate_process, forward_process)
    persist_path = Path.home() / ".llm_inference_platform.json"
    PersistInfo(job_id, port, node).dump(persist_path)
    atexit.register(lambda: persist_path.unlink())  # pylint: disable=unnecessary-lambda
    print("Press Ctrl + C once (!) to quit.")  # noqa: T201
    try:
        while True:
            if forward_process.poll() is not None:
                logger.critical("Port forwarding process died unexpectedly. Quitting")
                sys.exit(125)
            status_str, status = get_slurm_job_status(job_id)
            if status == JobState.RUNNING:
                pass
            elif status == JobState.COMPLETED:
                logger.info("Job completed successfully.")
                sys.exit(0)
            else:
                logger.critical("Job failed with status: %s", status_str)
                sys.exit(123)
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def deploy_cli(args: argparse.Namespace) -> None:
    """Run deployment from CLI"""
    deploy(
        model_name=args.name,
        revision=args.revision,
        model_dir=args.dir,
        quantization=args.quantization,
        context_length=args.context_length,
        extra_args=args.extra_args,
    )
