# local

Testing on local hardware (Alienware R15 X2 laptop).

# Development Notes

## Software Information

```bash
> nvidia-container-cli info
NVRM version:   525.125.06
CUDA version:   12.0

Device Index:   0
Device Minor:   0
Model:          NVIDIA GeForce RTX 3080 Ti Laptop GPU
Brand:          GeForce
GPU UUID:       GPU-bd3fa8f3-45af-3192-f030-7b9c4825eb29
Bus Location:   00000000:01:00.0
Architecture:   8.6

> nvidia-container-cli --version
cli-version: 1.13.5
lib-version: 1.13.5
build date: 2023-07-18T11:38+00:00
build revision: 66607bd046341f7aad7de80a9f022f122d1f2fce
build compiler: x86_64-linux-gnu-gcc-7 7.5.0
build platform: x86_64
build flags: -D_GNU_SOURCE -D_FORTIFY_SOURCE=2 -DNDEBUG -std=gnu11 -O2 -g -fdata-sections -ffunction-sections -fplan9-extensions -fstack-protector -fno-strict-aliasing -fvisibility=hidden -Wall -Wextra -Wcast-align -Wpointer-arith -Wmissing-prototypes -Wnonnull -Wwrite-strings -Wlogical-op -Wformat=2 -Wmissing-format-attribute -Winit-self -Wshadow -Wstrict-prototypes -Wunreachable-code -Wconversion -Wsign-conversion -Wno-unknown-warning-option -Wno-format-extra-args -Wno-gnu-alignof-expression -Wl,-zrelro -Wl,-znow -Wl,-zdefs -Wl,--gc-sections
```

## Testing 6 Sept 2023

Downloaded docker container using quoted example:

```bash
docker pull ghcr.io/huggingface/text-generation-inference:1.0.3
```

Initially deployment script [`run_llama-7b.sh`](./run_llama-7b.sh).

- Need to provide Huggingface Hub token for authentication.
- Initial model download takes time (shared model directory essential)
- No parameters crashed due to insufficient VRAM.
- Second attempt with following parameters deployed successfully:
  - Startup time was less than 30 seconds.

```bash
    [...]
    --quantize bitsandbytes \
    --max-batch-prefill-tokens=1024 \
    --max-total-tokens=2048 \
    [...]
```

Test query: 

```bash
$ curl 127.0.0.1:8080/generate \
             -X POST \
             -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
             -H 'Content-Type: application/json'
```

Response:

```json
{"generated_text":"\nWhat is Deep Learning? Deep learning is a subset of machine learning that is based on artificial neural"}⏎  
```

Logs:

```
2023-09-06T13:47:01.731233Z  INFO HTTP request{otel.name=POST /generate http.client_ip= http.flavor=1.1 http.host=127.0.0.1:8080 http.method=POST http.route=/generate http.scheme=HTTP http.target=/generate http.user_agent=curl/7.88.1 otel.kind=server trace_id=d6e2778955d17bf523e3dc568754ef3e}:generate{parameters=GenerateParameters { best_of: None, temperature: None, repetition_penalty: None, top_k: None, top_p: None, typical_p: None, do_sample: false, max_new_tokens: 20, return_full_text: None, stop: [], truncate: None, watermark: false, details: false, decoder_input_details: false, seed: None, top_n_tokens: None } total_time="986.532642ms" validation_time="681.374µs" queue_time="188.125µs" inference_time="985.66325ms" time_per_token="49.283162ms" seed="None"}: text_generation_router::server: router/src/server.rs:289: Success
```


Other notes:

- VRAM usage is very high: 15814MiB / 16384MiB (96.5%)
- Inference time 49ms / token