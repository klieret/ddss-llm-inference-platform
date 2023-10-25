# Usage

_WIP_: alpha version

The following steps should get you up and running with a `Llama-2-7b-chat-hf`
model running on Della and queryable from a local computer.

## 1. Connect to Della

If you know how to do this, then you can skip this step.

Otherwise, open a browser and go to [MyDella Cluster Shell Accesss](https://mydella.princeton.edu/pun/sys/shell/ssh/della8).

Note that if you are not on campus or your computer is not supported by the OIT security systems, you will need to first connect to the VPN.

See the [Knowledge Base article](https://princeton.service-now.com/service?id=kb_article&table=kb_knowledge&sys_id=ce2a27064f9ca20018ddd48e5210c745) for information on how to connect.


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

## 5. Optional: Forward Connection

If all goes well, you will see the following message:

```
Model deployed successfully. Here are your options to connect to the model:
1. If you are working on the della-gpu (head) node, no steps are necessary. Simply connect to localhost:<PORT>
2. If you are working somewhere else run the following command: 
    ssh -N -f -L localhost:8000:<NODE>:<PORT> <USERID>@della.princeton.edu
Afterwards, connect as in option 1.
```

If you are doing your work on the compute node, then you can simply 


Record the values of `<PORT>` and `<NODE>`. 


## 6. Create ssh tunnel from local computer

On your _own_ device (i.e., your laptop), run the following command to start an
ssh tunnel to the compute node, substituting:

- `<NODE>` with the output of the command used in Step 5
- `<USERID>` with your Princeton ID, e.g. `mj2976`.

```bash
ssh -N -f -L 8000:della-<NODE>:8000 <USERID>@della.princeton.edu
```

Note: _You may need to do some authentication steps here, depending on how you
connect to the HPC._

## 7. Use endpoint!

The endpoint is now accessible on your local computer! You can test it with the
following command:

```bash
curl localhost:8000/generate \
    -X POST \
    -d '{"inputs":"Hello World!","parameters":{"max_new_tokens":100, "repetition_penalty":2.5}}' \
    -H 'Content-Type: application/json'
```

## 8. Disconnecting/Cleaning Up

You can close the application by pressing `<Ctrl-C>` in the terminal where you
started the python command in Step 4.

You can close the local ssh tunnel by running:

```bash
ssh -O cancel -L 8000:della-<NODE>:8000 <USERID>@della.princeton.edu
```

## 9. Connect applications

_Next steps for us, but this should now work with most HuggingFace applications
by pointing to `localhost:8000`._
