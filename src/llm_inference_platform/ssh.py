import shlex
import socket
import subprocess
from typing import Any

from llm_inference_platform.utils.log import logger


def find_open_port() -> str:
    """Find open port that we can take for SSH port forwarding"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr = s.getsockname()
    s.close()
    return str(addr[1])


def forward_port(
    node: str, origin: str | int, target: str | int
) -> subprocess.Popen[Any]:
    """Forward port from local machine to remote machine"""
    cmd = ["ssh", "-N", "-L", f"{origin}:localhost:{target}", f"{node}"]
    logger.debug("Running command: %s", shlex.join(cmd))
    return subprocess.Popen(
        cmd,
    )  # pylint: disable=consider-using-with
