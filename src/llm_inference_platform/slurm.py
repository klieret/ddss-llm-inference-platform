from __future__ import annotations

import enum
import subprocess
import time
from enum import auto
from typing import Literal, Self

from utils.log import logger


class JobState(enum.Enum):
    """Slightly coarser categorization of SLURM job states."""

    RUNNING = auto()
    COMPLETED = auto()
    PENDING = auto()
    FAILED = auto()
    UNKNOWN = auto()

    def from_status_str(status: str) -> Self:
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
    logger.debug("Got status string '%s'", status_str_full)
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
    logger.debug("Got start time '%s'", start_time_str_full)
    return start_time_str_full.splitlines()[0].strip()


class WaitTillRunning:
    def __init__(self, job_id: str, *, poll_interval: int = 10):
        self._job_id = job_id
        self._poll_interval = poll_interval

    def user_feedback(
        self, message: str, *, level: Literal["info", "error"] = "info"
    ) -> None:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        else:
            msg = f"Unknown level {level}"
            raise ValueError(msg)

    def wait(self) -> None:
        self.user_feedback(f"Waiting for job {self._job_id} to start...")
        start_time = time.time()
        while True:
            status_str, status = get_slurm_job_status(sys.argv[1])
            logger.debug("Status: %s, %s", status_str, status)
            match status:
                case JobState.RUNNING:
                    self.user_feedback(f"Job {self._job_id} is running.")
                    return True
                case JobState.PENDING:
                    start_time = get_slurm_start_time(sys.argv[1])
                    self.user_feedback(
                        f"Job {self._job_id} is pending. Estimated start time {start_time}."
                    )
                    time.sleep(self._poll_interval)
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
                        time.sleep(self._poll_interval)
                    else:
                        self.user_feedback(
                            f"Job {self._job_id} status unknown. Please report this.",
                            level="error",
                        )
                        return False


if __name__ == "__main__":
    import sys

    wtr = WaitTillRunning(sys.argv[1])
    wtr.wait()
