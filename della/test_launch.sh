#!/bin/bash
# leaving this here in case we need this logic
# free_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# --model-id=$HF_HOME/models--meta-llama--Llama-2-70b-chat-hf/snapshots/9ff8b00464fc439a64bb374769dec3dd627be1c2 \

singularity run \
  --nv \
  --mount type=bind,src=$HF_HOME,dst=/data \
  --env HF_HOME=/data \
  --env HF_HUB_OFFLINE=1 \
  text-generation-inference_latest.sif \
  --model-id=/data/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235 \
  -p 8080
