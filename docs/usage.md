# Usage

_WIP_: alpha version

The following steps should get you up and running with a `Llama-2-7b-chat-hf` model running on Della and queryable from a local computer.


## 1. Connect to Della

_If you're in the alpha, we trust you know how to do this_.

## 2. Go to shared project directory

_There will be a better solution for here once we get the resources_.

```bash
cd /scratch/gpfs/mj2976/shared/llm-inference-platform
```

## 3. Activate environment

```bash
source .env/bin/activate
```

## 4. Run CLI tool

This python script populates and launches a SLURM job.

```bash
python src/llm_inference_platform/cli.py \
    deploy \
    --name models--meta-llama--Llama-2-7b-chat-hf \
    --revision 08751db2aca9bf2f7f80d2e516117a53d7450235 \
    --dir /scratch/gpfs/mj2976/shared/models
```

## 5. Get hostname of compute node

If all goes well, you will see the following message:

```
DEBUG: Got status string RUNNING
```

This means the server is now running.

You can get the hostname of the node it is running on using the `squeue` command.

Here's the one-liner:

```bash
squeue -u $USER -n llm-inference-platform -h -o '%N'
```

Record the output, e.g. `della-l07g4`


## 6. Create ssh tunnel from local computer

On your _own_ device (i.e., your laptop), run the following command to start an ssh tunnel to the compute node, substituting:

- `<NODE>` with the output of the command used in Step 5
- `<USERID>` with your Princeton ID, e.g. `mj2976`.

```bash
ssh -N -f -L 8000:della-<NODE>:8000 <USERID>@della.princeton.edu
```

Note: _You may need to do some authentication steps here, depending on how you connect to the HPC._


## 7. Use endpoint!

The endpoint is now accessible on your local computer! You can test it with the following command:

```bash
curl localhost:8000/generate \
    -X POST \
    -d '{"inputs":"Hello World!","parameters":{"max_new_tokens":100, "repetition_penalty":2.5}}' \
    -H 'Content-Type: application/json'
```

## 8. Disconnecting/Cleaning Up

You can close the application by pressing `<Ctrl-C>` in the terminal where you started the python command in Step 4.

You can close the local ssh tunnel by running:

```bash
ssh -O cancel -L 8000:della-<NODE>:8000 <USERID>@della.princeton.edu
```


## 9. Connect applications

_Next steps for us, but this should now work with most HuggingFace applications by pointing to `localhost:8000`._


