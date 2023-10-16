#!/usr/bin/env python
import os
import subprocess
from pathlib import Path
import gradio as gr
import logging
from typing import List


## Functions
def list_available_models(
    model_directory: os.PathLike = Path("./models"),
) -> list[os.PathLike]:
    model_list = sorted(Path(model_directory).iterdir())
    model_list = [m for m in model_list if m.is_dir()]
    return model_list


def construct_cmd(
    model_name: str | os.PathLike = "<SELECT MODEL>",
    model_dir: os.PathLike = Path("./models"),
    quantization: str = "None",
    context_length: int = 2048,
    container_type="docker",  # Docker or Singularity
) -> List[str]:
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
            raise ValueError(f"{container_type} is not supported.")

    # Arguments to text-generation-launcher
    cmd += [
        f"--max-total-tokens={context_length}",
        f"--quantization={quantization}",
        f"--model-name={model_name}",
    ]
    return cmd


def launch_container(cmd: List[str]) -> None:
    # Launch
    logging.info(" ".join())
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f"Launching process with PID {process.pid}")


def main():
    ## Global Arguments/Resources
    model_directory = Path.cwd() / "../local/data/"
    model_names = list_available_models(model_directory)
    quantization_types = ["None", "bitsandbytes", "GPTQ"]

    with gr.Blocks() as app:
        gr.Markdown("# Text Generation Launcher")

        # Interactable Objects
        modelDropdown = gr.Dropdown(choices=model_names)
        qtzDropdown = gr.Dropdown(choices=quantization_types)
        previewCmd = gr.Textbox(
            label="Command to evaluate",
            lines=11,
            value=" ".join(
                construct_cmd(
                    model_name=modelDropdown.value,
                    model_dir=model_directory,
                    quantization=qtzDropdown.value,
                )
            ),
        )
        previewBtn = gr.Button("Preview command")

        # Events
        modelDropdown.select(
            lambda choice: (
                choice,
                " ".join(
                    construct_cmd(
                        model_name=modelDropdown.value,
                        model_dir=model_directory,
                        quantization=qtzDropdown.value,
                    )
                ),
            ),
            inputs=modelDropdown,
            outputs=[modelDropdown, previewCmd],
        )

        qtzDropdown.select(
            lambda choice: (
                choice,
                " ".join(
                    construct_cmd(
                        model_name=modelDropdown.value,
                        model_dir=model_directory,
                        quantization=qtzDropdown.value,
                    )
                ),
            ),
            inputs=modelDropdown,
            outputs=[modelDropdown, previewCmd],
        )

        previewBtn.click(
            fn=lambda x: " ".join(
                construct_cmd(
                    model_name=modelDropdown.value,
                    model_dir=model_directory,
                    quantization=qtzDropdown.value,
                )
            ),
            inputs=[],
            outputs=[previewCmd],
        )

    return app


def test():
    app = main()
    app.launch(share=False)


test()

# if __name__=="__main__":
#     interface = main()
#     interface.launch()
