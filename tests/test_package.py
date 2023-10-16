from __future__ import annotations

import importlib.metadata

import llm_inference_platform as m


def test_version():
    assert importlib.metadata.version("llm_inference_platform") == m.__version__
