#!/usr/bin/env bash

#SBATCH --job-name=llm-inference-platform
#SBATCH --output=llm-inference-platform-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=250G
#SBATCH --constraint=gpu80

{% if email %}
#SBATCH --mail-type=fail
#SBATCH --mail-user={{email}}
{% endif %}

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

{{cmd}}
