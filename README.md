# Slurm vLLM Qwen launcher

Bash helpers and a Slurm batch script for launching a Qwen chat model with vLLM on a single GPU allocation. The script checks the runtime, chooses an available service port, optionally prefetches the model, starts the API server, and cleans up on exit.

## Requirements

- A Slurm cluster with the requested GPU resource and a compatible CUDA/vLLM environment.
- Bash, Conda, Python, `curl`, `vllm`, and `torch` available in the activated environment.
- Access to the model on Hugging Face if it is not already cached.

## Run

1. Copy `vllm.env.example` to `config/vllm.env` and adjust the resource and model settings. The live config is intentionally excluded from this repository.
2. Set `CONDA_SH_PATH` to the cluster's `conda.sh` path and set `REPO_DIR` if submitting from another directory.
3. Submit with `sbatch vllm_qwen_llm.sh`.

The batch file defaults to one `gpu:3g.20gb` resource, eight CPUs, 32 GB memory, and 24 hours. Confirm these values match your cluster before submitting. Runtime logs and job state are written below `run/` and are ignored.
