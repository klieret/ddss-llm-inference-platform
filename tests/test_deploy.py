import shlex
from pathlib import Path

from llm_inference_platform.deploy import (
    PersistInfo,
    construct_singularity_cmd,
    format_slurm_submission_script,
)


def test_format_slurm_submission_script():
    """Test `format_slurm_submission_script`"""
    cmd = ["echo", "hello world"]
    email = "test@testmail.com"
    script = format_slurm_submission_script(cmd, email)
    assert email in script
    assert shlex.join(cmd) in script
    assert "{{" not in script


def test_construct_singularity_cmd():
    sc = shlex.join(
        construct_singularity_cmd(
            model_name="test_model",
            revision="1",
        )
    )
    assert "--quantization" not in sc
    assert "--port=8000" in sc
    assert "test_model/snapshots/1" in sc


def test_persist_info(tmp_path: Path):
    pi = PersistInfo("123", "8000", "della-l01g02")
    tp = tmp_path / "test.json"
    pi.dump(tp)
    pi_recovered = PersistInfo.from_file(tp)
    assert pi == pi_recovered
