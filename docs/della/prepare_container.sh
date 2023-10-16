#!/usr/bin/env bash
module purge
module load cudatoolkit/12.2 cudnn/cuda-11.x/8.2.0
APPTAINER_CACHEDIR=/scratch/gpfs/$USER/APPTAINER_CACHE
APPTAINER_TMP=/tmp
singularity pull docker://ghcr.io/huggingface/text-generation-inference:latest # muhark: do we want to use latest or specific release?


# Download model?
