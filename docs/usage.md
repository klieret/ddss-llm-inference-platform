# Usage

```{warning}
This project is still at an early stage of development.
Breaking changes might occur.
```

## Local workflow

This workflow assumes that you have experience running commands in the command
line. If you are not sure what a command line is, you may prefer the Jupyterhub
workflow for now.

### Open terminal

Open a terminal on Della by either:

1. Open a terminal in the browser using
   [MyDella](https://mydella.princeton.edu/pun/sys/shell/ssh/della8)
2. Open a terminal locally and ssh to Della (i.e. `ssh della.princeton.edu`).

NB: Research Computing provides the following
[help article](https://researchcomputing.princeton.edu/support/knowledge-base/connect-ssh)
for connecting to the clusters via SSH.

### Install library locally

_The following workflow uses conda environment. You are welcome to use
virtualenv instead_.

1. Load the anaconda module

```bash
module purge
module load anaconda3/2023.3
```

2. Create a new conda environment:

```bash
conda create -n lip python
```

_I am naming it `lip` as short for `llm-inference-platform`. This is arbitrary._

3. Activate the new conda environment:

```bash
conda activate lip
```

4. Install the `llm-inference-platform` package

```bash
pip install git+https://github.com/princeton-ddss/llm-inference-platform.git
```

### (Optional) Download a model

During development you can use a model pre-downloaded from my (@muhark)
directory. In the future we aim to have a shared directory.

If you wish to use your own model, you can use the model download functionality.
The following command downloads
[bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m).

```bash
llm-inference-platform model-dl \
    --repo-id bigscience/bloom-560m \
    --revision main \
    --cache-dir=./models
```

### Run deploy command

The following command launches a SLURM job that runs a Singularity container
serving inference for the requested model. The example below has pre-populated
values for our initial test.

(If you downloaded a model in the previous step, you can change the parameters
accordingly).

```bash
llm-inference-platform deploy --name bigscience/bloom-560m
                              --revision main
                              --cache-dir /scratch/gpfs/mj2976/shared/models
```

If all goes well, you will get the following message in the output

```
[21:28:17 llmip] INFO: Model deployed successfully. Here are your options to connect to the model:
[21:28:17 llmip] INFO: 1. If you are working on the server running this scripts, no steps are necessary.
   Simply connect to localhost:38761.
[21:28:17 llmip] INFO: 2. If you are working somewhere else run the following command:
  ssh -N -f -L localhost:8000:della-l08g6:8000 mj2976@della.princeton.edu
   Afterwards, connect to localhost:8000
```

Copy the command under `INFO: 2.`:

```bash
  ssh -N -f -L localhost:8000:della-l08g6:8000 mj2976@della.princeton.edu
```

This command will forward the API serving the LLM from the compute node to your
local machine.

```{note}
You may need to do some authentication steps here, depending on how you
connect to the HPC.
```

## Use endpoint!

You can now use the endpoint via a Web API! You can test it with the following
command:

```bash
curl localhost:<PORT>/generate \
    -X POST \
    -d '{"inputs":"Hello World!","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

where `<PORT>` is either 8000, or whatever port you assigned in place of the
8000 in `localhost:8000` in the step above.

## Disconnecting/Cleaning Up

You can close the application by pressing `<Ctrl-C>` in the terminal where you
ran the deploy command. Please do not spam `<Ctrl-C>`.
