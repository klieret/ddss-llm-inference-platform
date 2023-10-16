# Della development notes

General workflow:

1. **Donwload TGI**: run `prepare_container.sh` on head node. This downloads and
   converts TGI into a Singularity image.
   - In future it would be better to have a shared use image.
2. **Download Model**: current solution is in `hf_model_downloader.py`
   - Issue: currently need to point to snapshot path
   - Python function in library parses directory to solve this, probably a
     better solution exists
3. **Launch SLURM session**: I use
   `salloc --nodes=1 --ntasks=1 --time=60:00 --cpus-per-gpu=12 --mem-per-gpu=250G --gres=gpu:2 --constraint=gpu80`
   for 7b Llama.
4. **Launch Container**: from compute node, `bash test_launch.sh`
5. **Query API**: you can use the following command, for example:

```
curl localhost:8080/generate \
    -X POST \
    -d '{"inputs":"Hello World!","parameters":{"max_new_tokens":100, "repetition_penalty":2.5}}' \
    -H 'Content-Type: application/json
```
