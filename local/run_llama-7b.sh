model=meta-llama/Llama-2-7b-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=$(cat ~/.cache/huggingface/token) # This is my token

docker run \
    --rm \
    --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -e HUGGING_FACE_HUB_TOKEN=$token \
    -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.3 \
    --quantize bitsandbytes \
    --max-batch-prefill-tokens=1024 \
    --max-total-tokens=2048 \
    --model-id $model
