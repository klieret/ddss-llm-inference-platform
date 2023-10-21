from llm_inference_platform.slurm import JobState


def test_job_state():
    js = JobState.from_status_str("RUNNING")
    assert js == JobState.RUNNING
