# LLM inference platform

[![CI](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/ci.yml)
[![Check Markdown links](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/check-links.yaml/badge.svg)](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/check-links.yaml)
[![Documentation Status](https://readthedocs.org/projects/princeton-llm-inference-platform/badge/?version=latest)](https://princeton-llm-inference-platform.readthedocs.io/en/latest/?badge=latest)
[![CD](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/cd.yml/badge.svg)](https://github.com/princeton-ddss/llm-inference-platform/actions/workflows/cd.yml)

<!-- SPHINX-START -->

Repository for tools to easily deploy LLMs on Princeton's della cluster.

## Current status

We provide an easy _command line interface_ to deploy LLMs (such as LLaMA) on
della. All functionality can be accessed installation-free in the browser using
the mydella JupyterHub interface.

## Usage

See [tutorial][].

[tutorial]:
  https://princeton-llm-inference-platform.readthedocs.io/en/latest/usage.html

## Vision & Roadmap

> It should be as easy for researchers to use an open-source LLM as a
> closed/proprietary LLM.

Read more about our [vision][] and [roadmap][].

[vision]:
  https://princeton-llm-inference-platform.readthedocs.io/en/latest/vision.html
[roadmap]:
  https://princeton-llm-inference-platform.readthedocs.io/en/latest/roadmap.html

## Development installation

Installation

```bash
pip3 install --editable '.[dev,test,docs]'
```

Please also install the pre-commit hook:

```bash
pipx run pre-commit install
```

Alternatively, you can run `pre-commit` manually with `nox -s lint`.

In addition, `nox` provides the following:

- To run addition python lint checks, run `nox -s pylint`
- To build the documentation, run `nox -s docs` (the resulting documentation
  will be rendered at `docs/_build/html/index.html`)
