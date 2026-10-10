# Excluded local files

These files were present in the source directory and were left out of the public repository:

- `config/vllm.env` (1.5 KB): cluster-specific runtime settings; excluded as a local environment config.
- `run/vllm_chat_19139.log` (about 62 KB): Slurm job log.
- `run/vllm_job_id` (6 bytes): generated job identifier.
- `vllm-qwen-llm_19139.log` (about 8 KB): Slurm job log.

No model weights or datasets were found among the files included in this project snapshot.
