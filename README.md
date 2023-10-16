# llm-inference-platform

Repo for developing hosted LLM inference solution on Princeton's Della cluster.

Developers:

- @muhark
- @klieret

## Solution

_In broad terms_

- LMs are hosted using HuggingFace Text Generation Inference container
- Users request configuration via web GUI (Gradio?), generates SLURM request
- TGI service forwarded back to GUI/API at user-facing web server

## Development Practices

- Where possible, task=issue=branch.
- No pushing directly to `main`.

## Components

### Inference Container

_Active Issues_:

- [#3](https://github.com/princeton-ddss/llm-inference-platform/issues/3)

We are using the
[`text-generation-inference`](https://github.com/huggingface/text-generation-inference)
(TGI) container from HuggingFace (HF).

This is their optimized production-grade solution for serving LLM inference in
their own products.

It primarily consists of two components:

- Web server (written in Rust): serves endpoints and manages request batching
- LM engine (Python): runs models compatible with the HF ecosystem, with various
  optimizations baked in.

HF provides this as a Docker container, which we are using via Singularity on
compute nodes.

### Web GUI

_Active Issues_:

- [#4](https://github.com/princeton-ddss/llm-inference-platform/issues/4)

Web GUI that basically exposes all of the TGI options via dropdown menus with
explanations of what they do/the trade-off they provide.

_Model Selection_:

Users will be able to select models from either a pre-designated list (stored in
a shared read-only path) or point the container to their own custom checkpoints.

_Resource Selection_:

For known architectures, optimal resource requests can be calculated with
testing and then used to populate the form.

### Web Server

_tbc_

- Access to compute nodes **necessary**.
- Outside access only from within VPN **highly desirable** (because this
  simplifies authentication issues).
