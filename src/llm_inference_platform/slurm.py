from __future__ import annotations

import enum
import shlex
import subprocess
import time
from enum import auto
from typing import Literal

from llm_inference_platform.utils.log import logger


class JobState(enum.Enum):
    """Slightly coarser categorization of SLURM job states."""

    RUNNING = auto()
    COMPLETED = auto()
    PENDING = auto()
    FAILED = auto()
    UNKNOWN = auto()

    @classmethod
    def from_status_str(cls, status: str) -> JobState:
        """Interpret output of SLURM job status command"""
        match status:
            case "RUNNING":
                return JobState.RUNNING
            case "COMPLETED" | "DEADLINE":
                return JobState.COMPLETED
            case "PENDING" | "REQUEUED":
                return JobState.PENDING
            case "FAILED" | "BOOT_FAIL" | "CANCELLED" | "NODE_FAIL" | "OUT_OF_MEMORY" | "PREEMPTED" | "TIMEOUT":
                return JobState.FAILED
            case _:
                return JobState.UNKNOWN


def get_slurm_job_status(job_id: str) -> tuple[str, JobState]:
    """Get the status of a job on the SLURM cluster.

    Args:
        job_id: The job ID of the job to get the status of.

    Returns:
        The status of the job as a string and as a `JobState` enum.
    """
    cmd = ["sacct", "-n", "-j", job_id, "--format=State", "-P"]
    logger.debug("Getting SLURM job status with '%s'", " ".join(cmd))
    status_str_full = subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).strip()
    logger.debug("Got status string '%r'", status_str_full)
    if not status_str_full:
        return "", JobState.UNKNOWN
    status_str = status_str_full.splitlines()[0].strip()
    return status_str, JobState.from_status_str(status_str)


def get_slurm_start_time(job_id: str) -> str:
    """Get the start time of a job on the SLURM cluster.

    Args:
        job_id: The job ID of the job to get the start time of.

    Returns:
        The start time of the job as a UNIX timestamp.
    """
    cmd = ["sacct", "-n", "-j", job_id, "--format=Start", "-P"]
    logger.debug("Getting SLURM start time with '%s'", " ".join(cmd))
    start_time_str_full = subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).strip()
    logger.debug("Got start time '%r'", start_time_str_full)
    return start_time_str_full.splitlines()[0].strip()


def get_slurm_node(job_id: str) -> str:
    """Get the node of a job on the SLURM cluster.

    Args:
        job_id: The job ID of the job to get the node of.

    Returns:
        The node of the job as a string.
    """
    cmd = ["squeue", "-j", job_id, "--noheader", "--format=%N"]
    logger.debug("Getting SLURM node with '%s'", shlex.join(cmd))
    node_str_full = subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).strip()
    logger.debug("Got node '%r'", node_str_full)
    return node_str_full.splitlines()[0].strip()


def cancel_slurm_job(job_id: str) -> None:
    """Cancel a job on the SLURM cluster.

    Args:
        job_id: The job ID of the job to cancel.
    """
    logger.warning("Cancelling SLURM job %s. PLEASE WAIT!", job_id)
    cmd = ["scancel", job_id]
    logger.debug("Canceling SLURM job with '%s'", shlex.join(cmd))
    subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).strip()
    logger.debug("Canceled job '%s'", job_id)


class WaitTillRunning:
    def __init__(self, job_id: str, *, poll_interval: int = 10):
        """Wait until a SLURM job is running

        Args:
            job_id: SLURM Job ID
            poll_interval: How often to poll the SLURM cluster for job status
        """
        self._job_id = job_id
        self._poll_interval = poll_interval

    def user_feedback(
        self, message: str, *, level: Literal["info", "error"] = "info"
    ) -> None:
        """Log a message or update a dialogue box."""
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        else:
            msg = f"Unknown level {level}"  # type: ignore[unreachable]
            raise ValueError(msg)

    def wait(self) -> bool:
        """Wait until job is running

        Returns:
            true if job is running, false otherwise
        """
        self.user_feedback(f"Waiting for job {self._job_id} to start...")
        start_time = time.time()
        iter_running = 0
        while True:
            status_str, status = get_slurm_job_status(self._job_id)
            logger.debug("Status: %s, %s", status_str, status)
            match status:
                case JobState.RUNNING:
                    if iter_running == 0:
                        self.user_feedback(
                            f"Job {self._job_id} is running. Will wait one more iteration to make "
                            "sure it doesn't immediately fail."
                        )
                        iter_running += 1
                    else:
                        return True
                case JobState.PENDING:
                    expected_start_time = get_slurm_start_time(self._job_id)
                    self.user_feedback(
                        f"Job {self._job_id} is pending. Estimated start time {expected_start_time}."
                    )
                case JobState.FAILED:
                    self.user_feedback(f"Job {self._job_id} failed.", level="error")
                    return False
                case JobState.COMPLETED:
                    self.user_feedback(
                        f"Job {self._job_id} already completed. Please start a new one."
                    )
                    return False
                case JobState.UNKNOWN:
                    if time.time() - start_time < 30:
                        self.user_feedback(
                            f"Job {self._job_id} status unknown. Please wait a bit longer.",
                            level="info",
                        )
                    else:
                        self.user_feedback(
                            f"Job {self._job_id} status unknown. Please report this.",
                            level="error",
                        )
                        return False
            time.sleep(self._poll_interval)


if __name__ == "__main__":
    import sys

    wtr = WaitTillRunning(sys.argv[1])
    wtr.wait()
