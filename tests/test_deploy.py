import shlex

from llm_inference_platform.deploy import format_slurm_submission_script


def test_format_slurm_submission_script():
    """Test `format_slurm_submission_script`"""
    cmd = ["echo", "hello world"]
    email = "test@testmail.com"
    script = format_slurm_submission_script(cmd, email)
    assert email in script
    assert shlex.join(cmd) in script
    assert "{{" not in script
