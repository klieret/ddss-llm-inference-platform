#!/usr/bin/env python3

"""Submit container with a web interface"""

import argparse
from pathlib import Path

import gradio as gr

from llm_inference_platform.deploy import construct_cmd, list_available_models


def get_cli() -> argparse.ArgumentParser:
    """Get argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-directory",
        default=Path.cwd() / "../local/data/",
        type=Path,
    )
    return parser


def build_gradio_app(*, model_directory: Path) -> gr.Blocks:
    """Start the Gradio app."""
    assert model_directory.is_dir()
    ## Global Arguments/Resources
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


def main() -> None:
    """Run the Gradio app from the CLI."""
    parser = get_cli()
    args = parser.parse_args()
    app = build_gradio_app(model_directory=args.model_directory)
    app.launch(share=False)


if __name__ == "__main__":
    main()
