# Usage

_The following instructions explain how to use the inference deployment tool
without installing anything on your own._ _It is still in early alpha meaning
breaking changes to usage and interface are guaranteed._

The following steps should get you up and running with a `Llama-2-7b-chat-hf`
model running on Della and queryable from a local computer.

## No-Install Method

### 1. Connect to Della

If you know how to do this, then you can skip this step.

Otherwise, open a browser and go to
[MyDella Cluster Shell Access](https://mydella.princeton.edu/pun/sys/shell/ssh/della8).

Note that if you are not on campus or your computer is not supported by the OIT
security systems, you will need to first connect to the VPN.

See the
[Knowledge Base article](https://princeton.service-now.com/service?id=kb_article&table=kb_knowledge&sys_id=ce2a27064f9ca20018ddd48e5210c745)
for information on how to connect.

### 2. Activate environment

From the terminal connected to Dela, run the following command:

```bash
source /scratch/gpfs/mj2976/shared/llm-inference-platform/.env/bin/activate
```

This command sets your Python environment to the one that has our tools
pre-loaded. You can check that whether the command has worked with
`which python`, which should output the path above with `python` instead of
`activate`.

### 3. Run deploy command

The following command launches a SLURM job that runs a Singularity container
serving inference for the requested model. The example below has pre-populated
values for our initial test.

```bash
llm-inference-platform deploy \
    --name meta-llama/Llama-7b-chat-hf \
```

### 4. Optional: Forward Connection

If all goes well, you will see the following message:

```
Model deployed successfully. Here are your options to connect to the model:
1. If you are working on the della (head) node, no steps are necessary. Simply connect to localhost:<NODE>.
2. If you are working somewhere else run the following command:
ssh -N -f -L localhost:8000:<NODE>:8000 <USERID>@della.princeton.edu
   Afterwards, connect as in option 1.
```

If you are doing your work on the compute node, then you can proceed to the next
step.

If you want to use the model from your own computer, open a terminal on your own
device.

Run the following command to start an ssh tunnel to the compute node,
substituting:

- `<NODE>` the compute node name in the message above
- `<USERID>` with your Princeton ID, e.g. `mj2976`.

```bash
ssh -N -f -L 8000:<NODE>:8000 <USERID>@della.princeton.edu
```

Note: _You may need to do some authentication steps here, depending on how you
connect to the HPC._

## 7. Use endpoint!

You can now use the endpoint via a Web API! You can test it with the following
command:

If you are on a compute node:

```bash
curl localhost:<NODE>/generate \
    -X POST \
    -d '{"inputs":"Hello World!","parameters":{"max_new_tokens":100, "repetition_penalty":2.5}}' \
    -H 'Content-Type: application/json'
```

If you are on your own computer and ran Step 4:

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
ssh -O cancel -L 8000:<NODE>:8000 <USERID>@della.princeton.edu
```

## 9. Connect applications

_Next steps for us, but this should now work with most HuggingFace applications
by pointing to `localhost:8000`._
