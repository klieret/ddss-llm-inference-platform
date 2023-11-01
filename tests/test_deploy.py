import shlex
from pathlib import Path

from llm_inference_platform.deploy import (
    _construct_singularity_cmd,
    _format_slurm_submission_script,
    _PersistInfo,
)


def test_format_slurm_submission_script():
    """Test `format_slurm_submission_script`"""
    cmd = ["echo", "hello world"]
    email = "test@testmail.com"
    script = _format_slurm_submission_script(cmd, email)
    assert email in script
    assert shlex.join(cmd) in script
    assert "{{" not in script


def test_construct_singularity_cmd(tmp_path: Path):
    sc = shlex.join(
        _construct_singularity_cmd(
            weight_dir=tmp_path,
        )
    )
    assert "--quantization" not in sc
    assert "--port=8000" in sc


def test_persist_info(tmp_path: Path):
    pi = _PersistInfo("123", "8000", "della-l01g02")
    tp = tmp_path / "test.json"
    pi.dump(tp)
    pi_recovered = _PersistInfo.from_file(tp)
    assert pi == pi_recovered
