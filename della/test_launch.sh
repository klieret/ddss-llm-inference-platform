#!/bin/bash
# singularity run \
#     --bind $HF_HOME \
#     text-generation-inference_latest.sif \
#     --model-id /data/models--google--flan-ul2 \
#     --port 8896 
  # --model-id=meta-llama/Llama-2-7b-hf \
  # --revision=main \
  # --model-id=$HF_HOME/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9 \

free_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

singularity run \
  --nv \
  --mount type=bind,src=$HF_HOME,dst=/data/hub \
  --env HF_HOME=/data/hub \
  --env HF_HUB_OFFLINE=1 \
  text-generation-inference_latest.sif \
  --huggingface-hub-cache=/data/hub \
  --model-id=$HF_HOME/models--meta-llama--Llama-2-70b-hf/snapshots/cc8aa03a000ff08b4d5c5b39673321a2a396c396 \
  -p 8080
