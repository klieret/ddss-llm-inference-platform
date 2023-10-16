#!/usr/bin/env python
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import gradio as gr

# fixme
# pylint: disable=missing-function-docstring


def list_available_models(
    model_directory: os.PathLike = Path("./models"),
) -> list[os.PathLike]:
    model_list = sorted(Path(model_directory).iterdir())
    return [m for m in model_list if m.is_dir()]


def construct_cmd(
    model_name: str | os.PathLike = "<SELECT MODEL>",
    model_dir: os.PathLike = Path("./models"),
    quantization: str = "None",
    context_length: int = 2048,
    container_type="docker",  # Docker or Singularity
) -> list[str]:
    # Construct argument string
    cmd = []
    # Containerization arguments
    match container_type:
        case "docker":
            cmd += [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--shm-size",
                "1g",
                "-v",
                f"{model_dir.absolute()}:/data",
                "ghcr.io/huggingface/text-generation-inference:latest",
            ]
        case "singularity":
            cmd += [
                "singularity",
                "run",
                "--nv",
                "--mount",
                f"type=bind,src={model_dir.absolute()},dst=/data",
                "--env",
                "HF_HOME=/data",
                "--env",
                "HF_HUB_OFFLINE=1",
                "text-generation-inference_latest.sif",
            ]
        case _:
            msg = f"{container_type} is not supported."
            raise ValueError(msg)

    # Arguments to text-generation-launcher
    cmd += [
        f"--max-total-tokens={context_length}",
        f"--quantization={quantization}",
        f"--model-name={model_name}",
    ]
    return cmd


def launch_container(cmd: list[str]) -> None:
    # Launch
    logging.info("Running %s", " ".join(cmd))
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logging.info("Launching process with PID %s", process.pid)


def main():
    ## Global Arguments/Resources
    model_directory = Path.cwd() / "../local/data/"
    model_names = list_available_models(model_directory)
    quantization_types = ["None", "bitsandbytes", "GPTQ"]

    with gr.Blocks() as app:
        gr.Markdown("# Text Generation Launcher")

        # Interactable Objects
        model_dropdown = gr.Dropdown(choices=model_names)
        qtz_dropdown = gr.Dropdown(choices=quantization_types)
        preview_cmd = gr.Textbox(
            label="Command to evaluate",
            lines=11,
            value=" ".join(
                construct_cmd(
                    model_name=model_dropdown.value,
                    model_dir=model_directory,
                    quantization=qtz_dropdown.value,
                )
            ),
        )
        preview_btn = gr.Button("Preview command")

        # Events
        model_dropdown.select(
            lambda choice: (
                choice,
                " ".join(
                    construct_cmd(
                        model_name=model_dropdown.value,
                        model_dir=model_directory,
                        quantization=qtz_dropdown.value,
                    )
                ),
            ),
            inputs=model_dropdown,
            outputs=[model_dropdown, preview_cmd],
        )

        qtz_dropdown.select(
            lambda choice: (
                choice,
                " ".join(
                    construct_cmd(
                        model_name=model_dropdown.value,
                        model_dir=model_directory,
                        quantization=qtz_dropdown.value,
                    )
                ),
            ),
            inputs=model_dropdown,
            outputs=[model_dropdown, preview_cmd],
        )

        preview_btn.click(
            fn=lambda _: " ".join(
                construct_cmd(
                    model_name=model_dropdown.value,
                    model_dir=model_directory,
                    quantization=qtz_dropdown.value,
                )
            ),
            inputs=[],
            outputs=[preview_cmd],
        )

    return app


def test():
    app = main()
    app.launch(share=False)


test()

# if __name__=="__main__":
#     interface = main()
#     interface.launch()
