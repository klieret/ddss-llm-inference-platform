#!/usr/bin/env python
import os
import gradio as gr


## Functions
def get_model_names(model_directory):
    # A simple function to get all model names (subdirectories) from a given directory
    # Assuming every subdirectory in model_directory is a separate model
    return os.listdir(model_directory)


def launch_huggingface_job(model_name, quantization, context_length):
    # Placeholder function: this is where you'd put logic to launch the HuggingFace job.
    return f"Launching {model_name} with {quantization} and context length {context_length}!"


## Global Arguments/Resources
model_directory = "./models"  # Replace this with your model directory
model_names = get_model_names(model_directory)
quantization_types = ["None", "bitsandbytes", "GPTQ"]


## Interface
interface = gr.Interface(
    fn=launch_huggingface_job,
    inputs=[
        gr.inputs.Dropdown(label="Model", choices=model_names),
        gr.inputs.Dropdown(
            label="Quantization", choices=quantization_types, default="None"
        ),
        gr.inputs.Slider(
            label="Context Length", min_value=10, max_value=1000, step=1, default=512
        ),
    ],
    outputs=gr.outputs.Textbox(),
    live=True,  # Use 'live' if you want real-time feedback, else remove this line.
)

interface.launch()
