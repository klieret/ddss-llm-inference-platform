# Vision

Our vision is to facilitate the use of "open" language models in academic research. In a sentence, the guiding principle is:

> It should be as easy for researchers to use an open-source LLM as a closed/proprietary LLM.

We propose to do this by creating a _simple_, _code-free_, _graphical interface_ for the following:

- **Deploying** open LMs on university HPC infrastructure
- **Interacting** with open LMs via familiar graphical interfaces (chat, completion, etc.)

We believe that there are considerable benefits to creating this product:

- **Enabling** a broad range of researchers to engage with LM technology.
- **Facilitating** open, reproducible research with LMs.
- Creating a means for **optimizing** shared resource usage.

## Targets

The current idea is to break down development into three stages.

- [ ] CLI Deployment
- [ ] API Interaction
- [ ] GUI Deployment
- [ ] GUI Interaction


### Stage 1: Deployment Interface

The first step is to establish a GUI for selecting and deploying LMs on Princeton's HPC. We currently propose to do this within the SLURM/queueing model of resource allocation, where users can request up to 1 hour of inference time.

The deployment interface can be made with [Open OnDemand](https://openondemand.org/) or a simple web interface 










