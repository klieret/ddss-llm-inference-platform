from llm_inference_platform.ssh import find_open_port, forward_port


def test_find_open_port() -> None:
    """Test `find_open_port`"""
    port = find_open_port()
    assert isinstance(port, str)
    assert int(port) > 0


def test_forward_port(fp) -> None:
    """Test `forward_port`"""
    fp.register(["ssh", "-N", "-L", "8000:localhost:8000", "della-l01g02"])
    forward_port("della-l01g02", 8000, 8000)
