import pytest

from llm_inference_platform._slurm import (
    JobState,
    WaitTillRunning,
    cancel_slurm_job,
    get_slurm_job_status,
    get_slurm_node,
    get_slurm_start_time,
    sbatch,
)


def test_job_state():
    js = JobState.from_status_str("RUNNING")
    assert js == JobState.RUNNING


def test_get_slurm_job_status(fp):
    stdout = "RUNNING\nRUNNING\nRUNNINGJKJ"
    fp.register(["sacct", "-n", "-j", "12345", "--format=State", "-P"], stdout=stdout)
    status, js = get_slurm_job_status("12345")
    assert status == "RUNNING"
    assert js == JobState.RUNNING


def test_get_slurm_start_time(fp):
    stdout = "2021-10-05T17:05:00\nasdf"
    fp.register(["sacct", "-n", "-j", "12345", "--format=Start", "-P"], stdout=stdout)
    start_time = get_slurm_start_time("12345")
    assert start_time == "2021-10-05T17:05:00"


def test_get_slurm_node(fp):
    stdout = "della-l01g02\nasdf"
    fp.register(["squeue", "-j", "12345", "--noheader", "--format=%N"], stdout=stdout)
    node = get_slurm_node("12345")
    assert node == "della-l01g02"


def test_cancel_slurm_job(fp):
    fp.register(["scancel", "12345"])
    cancel_slurm_job("12345")


def test_sbatch(fp):
    fp.register(["sbatch", fp.any()], stdout="Submitted batch job 12345")
    job_id = sbatch("test.sh")
    assert job_id == "12345"


def test_sbatch_failed(fp):
    fp.register(["sbatch", fp.any()], stdout="asdf")
    with pytest.raises(RuntimeError, match=".*Could not submit.*"):
        sbatch("test.sh")


def test_wtr(fp):
    # Need to register it twice, because it's polled twice
    fp.register(["sacct", fp.any()], stdout="RUNNING\nasdf")
    fp.register(["sacct", fp.any()], stdout="RUNNING\nasdf")
    wtr = WaitTillRunning("123", poll_interval=0)
    wtr.wait()


def test_wtr_pend_first(fp):
    fp.register(["sacct", fp.any()], stdout="UNKNOWN\nasdf")
    fp.register(["sacct", fp.any()], stdout="PENDING\nasdf")
    fp.register(["sacct", fp.any(), "--format=Start", "-P"], stdout="UNKNOWN\nasdf")
    fp.register(["sacct", fp.any()], stdout="RUNNING\nasdf")
    fp.register(["sacct", fp.any()], stdout="RUNNING\nasdf")
    wtr = WaitTillRunning("123", poll_interval=0)
    wtr.wait()
