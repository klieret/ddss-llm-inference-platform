from __future__ import annotations

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

from llm_inference_platform._slurm import (
    JobState,
    WaitTillRunning,
    cancel_slurm_job,
    get_slurm_job_status,
    get_slurm_node,
    sbatch,
)
from llm_inference_platform._ssh import find_open_port, forward_port
from llm_inference_platform.hf_model_downloader import HF_DEFAULT_HOME
from llm_inference_platform.utils.log import DEFAULT_LOGGER_PATH, logger

# MH: Path used during development; will substitute once shared
# resources allocated
SHARED_RESOURCE_DIR = Path("/scratch/gpfs/mj2976/shared")


def list_available_models(
    model_directory: str | os.PathLike[Any] = Path("./models"),
) -> list[Path]:
    """Given a model directory, list all model subdirectories"""
    model_list = sorted(Path(model_directory).iterdir())
    return [m for m in model_list if m.is_dir()]


def _construct_docker_cmd(
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


def _construct_singularity_cmd(
    *,
    weight_dir: Path,
    quantization: str | None = None,
    context_length: int = 2048,
    singularity_image: Path = Path("./text-generation-inference_latest.sif"),
    extra_args: list[str] | None = None,
) -> list[str]:
    """Run ``text-generation-inference`` in singularity container

    Args:
        weight_dir: Path to model weights
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
    model_dir = weight_dir.resolve().parent.parent
    model_dir_mount = Path("/") / "data"
    weight_dir_mount = model_dir_mount / weight_dir.resolve().relative_to(model_dir)
    cmd = [
        "singularity",
        "run",
        "--nv",
        "--mount",
        f"type=bind,src={model_dir},dst={model_dir_mount}",
        "--env",
        "HF_HOME=/data",
        "--env",
        "HF_HUB_OFFLINE=1",
        f"{singularity_image.resolve()}",
        f"--max-total-tokens={context_length}",
        f"--model-id={weight_dir_mount}",
        "--env",
        "--port=8000",
        *extra_args,
    ]
    if quantization is not None:
        cmd.append(f"--quantize={quantization}")
    return cmd


def _format_slurm_submission_script(cmd: list[str], email: str = "") -> str:
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


class _PersistInfo(NamedTuple):
    """Information to write to a file in user's home directory"""

    job_id: str
    port: str
    node: str

    def dump(self, path: Path) -> None:
        """Dump information to a file"""
        with path.open("w") as f:
            json.dump(self._asdict(), f)  # pylint: disable=no-member

    @classmethod
    def from_file(cls, path: Path) -> _PersistInfo:
        """Load information from a file"""
        with path.open("r") as f:
            dct = json.load(f)
        return cls(**dct)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Terminate process and log it"""
    logger.debug("Terminating process with PID %s", process.pid)
    process.terminate()


def _print_usage_instructions(port: str, node: str) -> None:
    """Tell user what SSH command to run on their own machine"""
    logger.info(
        "Model deployed successfully. Here are your options to connect to the model:"
    )
    logger.info(
        "1. If you are working on the server running this scripts, no steps are necessary.\n"
        "   Simply connect to localhost:%s.",
        port,
    )
    user_id = os.environ.get("USER")
    logger.info(
        "2. If you are working somewhere else run the following command:\n"
        "  ssh -N -f -L localhost:8000:%s:8000 %s@della.princeton.edu\n"
        "   Afterwards, connect to localhost:8000",
        node,
        user_id,
    )
    logger.info("Press Ctrl + C once (!) to quit.")


def _print_debug_information(job_id: str | None = None) -> None:
    """Print debug information at the end of the script"""
    logger.warning(
        "If this script failed or did not work as expected, please include the "
        "debug output in your report. It is saved in the file: %s.",
        DEFAULT_LOGGER_PATH,
    )
    if job_id is not None:
        log_file = Path(f"llm-inference-platform-{job_id}.log")
        if log_file.is_file():
            logger.warning(
                "Please also include the SLURM log file for job: %s",
                log_file,
            )


def deploy(**kwargs) -> None:  # type: ignore[no-untyped-def]
    """Deploy a model to the cluster.

    Args:
        See `construct_singularity_cmd` for arguments.
    """
    atexit.register(_print_debug_information)
    cmd = _construct_singularity_cmd(**kwargs)
    logger.debug("Singularity command: %s", shlex.join(cmd))
    script = _format_slurm_submission_script(cmd)
    job_id = sbatch(script)
    # Doesn't matter if slurm job is already dead before we call this
    # cleanup, but need to make sure it never survives
    atexit.register(cancel_slurm_job, job_id)
    wtr = WaitTillRunning(job_id)
    atexit.unregister(_print_debug_information)
    atexit.register(_print_debug_information, job_id)
    success = wtr.wait()
    if not success:
        msg = "Job failed to start, check SLURM log for details."
        logger.critical(msg)
        sys.exit(234)
    port = find_open_port()
    node = get_slurm_node(job_id)
    logger.info("Forwarding port 8000 on %s to localhost:%s", node, port)
    forward_process = forward_port(node, port, 8000)
    atexit.register(_terminate_process, forward_process)
    persist_path = Path.home() / ".llm_inference_platform.json"
    _PersistInfo(job_id, port, node).dump(persist_path)
    atexit.register(lambda: persist_path.unlink())  # pylint: disable=unnecessary-lambda
    _print_usage_instructions(port, node)
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


def get_weight_dir(
    model_ref: str,
    *,
    model_dir: str | os.PathLike[Any] = HF_DEFAULT_HOME,
    revision: str = "main",
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir
