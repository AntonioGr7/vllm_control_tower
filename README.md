# vLLM Control Tower

A terminal control panel for [vLLM](https://github.com/vllm-project/vllm).

- Define deployments as YAML files in a `configs/` folder: one or more vLLM models served together.
- Browse them in a TUI with the arrow keys; serve, stop and monitor them live.
- Before you serve, estimate VRAM use, KV cache capacity and tokens/s on your GPU, or on any GPU from the spec table (H100, A100, L4, ...).

![Estimate tab: VRAM breakdown, KV cache capacity and decode throughput for Qwen3-0.6B-FP8 on a 4 GB RTX 2050](docs/tui-estimate.png)

## Install

```bash
uv sync                  # the control tower itself
uv sync --extra vllm     # also install vLLM into the same venv (large download)
```

vLLM can also live elsewhere: set a model's `vllm.executable` to any `vllm` binary.
Executables named without a path are looked up first in this project's `.venv/bin`, then on `PATH`.

## Use

```bash
uv run vct                       # open the TUI
uv run vct list                  # deployments and their state
uv run vct estimate qwen3-0.6b   # memory / KV cache / throughput estimate on the local GPU
uv run vct estimate qwen3-8b-h100 --gpu H100          # ...or on hardware you don't have
uv run vct start qwen3-0.6b      # serve in the background, wait for /health
uv run vct bench                 # TUI in bench mode: deployments you start get benchmarked
uv run vct bench qwen3-0.6b      # headless: start if needed, measure, compare, stop
uv run vct status                # running servers + GPU usage
uv run vct logs qwen3-0.6b -f    # a multi-model deployment needs the model: duo-chat-embed.chat
uv run vct stop qwen3-0.6b       # or: vct stop --all
```

### TUI keys

| key | action |
| --- | --- |
| `↑` `↓` | select a deployment or one of its models |
| `←` `→` / `1`–`4` | switch tab: Overview, Estimate, Monitor, Logs |
| `s` / `enter` | serve the selection; a deployment starts its models in order |
| `x` | stop it (asks for confirmation) |
| `b` | benchmark the selection (it must be running) |
| `h` | choose the GPU to estimate for |
| `r` | reload deployments from disk |
| `q` | quit (running servers keep running) |

## Deployment format

Every file is a deployment: a name, shared defaults and a `models:` list. A single model is a deployment with one entry.

```yaml
name: qwen3-0.6b               # defaults to the file name
description: Qwen3 0.6B on a 4 GB GPU
host: 127.0.0.1                # default for every model
env:                           # shared; a model's own env wins. ${VAR} and ${VAR:-default} are expanded
  HF_TOKEN: ${HF_TOKEN}
models:
  - name: main                 # optional: defaults to "main" (or m0, m1, ... when there are several)
    model: Qwen/Qwen3-0.6B     # HF repo id or local path
    port: 8000
    vllm:
      executable: vllm
      # image: vllm/vllm-openai:latest   # run `vllm serve` in this Docker image instead
      args:                    # any `vllm serve` option, with or without leading dashes
        gpu-memory-utilization: 0.8
        max-model-len: 4096
        max-num-seqs: 16
      flags: [enforce-eager]   # boolean switches
      extra_args: []           # passed through verbatim
    estimate:                  # workload assumed by the throughput estimate
      avg_prompt_tokens: 512
      avg_output_tokens: 256
```

Each model is one vLLM server, named `<deployment>.<model>` (e.g. `qwen3-0.6b.main`). That name is what you see in `vct status`, and it names the log file.

### Several models in one deployment

[configs/duo-chat-embed.yaml](configs/duo-chat-embed.yaml) serves a chat model and an embedding model on the same 4 GB GPU:

```yaml
name: duo-chat-embed
env: {HF_TOKEN: ${HF_TOKEN}}
models:
  - name: chat                    # duo-chat-embed.chat
    model: Qwen/Qwen3-0.6B-FP8
    port: 8010
    vllm: {args: {gpu-memory-utilization: 0.55, max-model-len: 4096}}
  - name: embed                   # duo-chat-embed.embed
    model: BAAI/bge-small-en-v1.5
    port: 8011
    vllm: {args: {runner: pooling, gpu-memory-utilization: 0.18, max-model-len: 512}}
```

- `vct start duo-chat-embed` starts models in the order listed. Each one starts only after the previous one passes its health check, because vLLM sizes its KV cache from the free memory it sees at startup.
- `vct stop duo-chat-embed` stops them in reverse order.
- `vct estimate duo-chat-embed` adds a combined check. Each vLLM instance claims `gpu-memory-utilization × total VRAM`, so models on the same GPU must sum to less than what is free, minus one CUDA context (~0.2 GiB) per model. Models with different `CUDA_VISIBLE_DEVICES` are budgeted separately.
- One model can be targeted on its own: `vct stop duo-chat-embed.embed`, `vct logs duo-chat-embed.chat -f`. In the TUI and `vct list`, every deployment lists its models underneath, even when there is only one. The deployment row acts on all of them, and its Monitor and Logs tabs combine them. A model row acts on that model alone.

![Deployment estimate: a chat model and an embedding model sharing one GPU, with the combined VRAM budget per model](docs/tui-group.png)

### Docker images

Set `vllm.image` and the model runs from that image instead of a local vLLM. Everything else in the file stays the same: `args`, `flags`, `env` and the port mean what they mean for a local process, and the TUI, logs, monitor and benchmark work unchanged.

```yaml
    vllm:
      image: vllm/vllm-openai:qwen38    # or an NGC image such as nvcr.io/nvidia/vllm:26.05-py3
      args: {gpu-memory-utilization: 0.7, max-model-len: 32768}
      docker:                           # optional
        hf_cache: /data/huggingface     # default: $HF_HOME or ~/.cache/huggingface
        args: [--shm-size, 16g]         # extra `docker run` options, verbatim
```

`vct` runs `docker run --rm --init --name vct-<deployment>.<model> --gpus all --ipc host --network host`. Host networking keeps `host:` and `port:` working as for a local process. It also:

- mounts the Hugging Face cache, so the container reuses your downloads and the estimator reads the same files. A `model:` that is an absolute local path is mounted read-only at the same path;
- passes each `env:` variable by name only (`-e HF_TOKEN`), so values never appear in the command line or the log;
- forces the entrypoint to `vllm`, so NGC images and `vllm/vllm-openai` images work alike;
- on `vct stop`, runs `docker stop` on the container. On `vct start`, it removes a leftover container with the same name, e.g. one whose `docker run` was killed.

Docker needs the NVIDIA Container Toolkit, and your user must be in the `docker` group. Pull the image first (`docker pull <image>`): the first `vct start` would otherwise pull it as part of startup, and multi-GB images can exceed the 15-minute health wait. The container runs as root, so files it adds to the Hugging Face cache are owned by root.

### DGX Spark

[NVIDIA's guide](https://build.nvidia.com/spark/vllm/instructions) serves vLLM from containers. Both ways work with `vct`:

- **Local vLLM:** `uv sync --extra vllm` installs aarch64 wheels with CUDA, and vLLM supports the GB10 (`sm_121`) natively. Check with `uv run python -c "import torch; print(torch.cuda.get_device_capability())"`, which should print `(12, 1)`. [configs/spark-chat-embed.yaml](configs/spark-chat-embed.yaml) uses this.
- **NVIDIA's images:** set `vllm.image` to the image NVIDIA lists for the model. New models and NVFP4 checkpoints often run there before a vLLM release supports them. [configs/spark-qwen38.yaml](configs/spark-qwen38.yaml) uses this.

One-time setup on the Spark: `docker ps` must work without `sudo` (`sudo usermod -aG docker $USER && newgrp docker`), and gated models need `HF_TOKEN` exported or `hf auth login`.

The GB10 has no dedicated VRAM, and NVML does not report memory for it, so `vct` budgets against system RAM from `/proc/meminfo`. Page cache counts as free memory, the same way vLLM's own startup check counts it on integrated GPUs. `gpu-memory-utilization` is then a share of memory the OS also uses, so keep the models' sum well below 1. Estimate a Spark deployment from any machine with `vct estimate spark-chat-embed --gpu GB10`.

Runtime state lives in `.vct/`: PID files, per-server logs (the previous run is kept as `.log.prev`), and cached model metadata. Override the locations with `VCT_CONFIG_DIR` and `VCT_RUNTIME_DIR`.

## Benchmark: measured vs estimated

`uv run vct bench` opens the same TUI in **bench mode**. Start a deployment with `s` as usual. As soon as all its models are healthy, vct benchmarks them one by one and shows the results at the top of the Estimate tab. The models keep running afterwards, and the Monitor tab shows the benchmark traffic live. In normal mode, `b` benchmarks whatever is selected and running.

For each model the benchmark:

1. sends a warm-up request, then streamed requests at the same batch sizes as the estimate (1, 2, 4, 8 … up to `--max-concurrency`, `max-num-seqs` and what vLLM's KV cache holds). Every request uses the deployment's prompt length and generates exactly `--output` tokens (default: the configured output length, at most 128). Embedding models (`runner: pooling`) get `/v1/embeddings` requests instead;
2. reports measured tok/s, per-request speed and time to first token next to the estimate, plus the memory vLLM actually allocated (read from its startup log);
3. saves the result under `.vct/bench/` and calibrates the throughput model for this GPU and mode (eager vs CUDA graphs). Later estimates on this GPU include the measured per-step overhead and say so.

`vct bench <deployment>` does the same without the TUI: it starts the deployment if needed, prints the comparison and stops what it started (`--keep` leaves it running).

`vct estimate` and the Estimate tab keep showing the last benchmark for each model, as long as its config hasn't changed since. Prompts are unique token sequences, so vLLM's prefix cache can't inflate the numbers. A benchmark loads the server for a minute or two, so don't run it against one that's serving real traffic.

## How the estimate works

Model facts come from `config.json` and the safetensors headers. They are read from the local Hugging Face cache when available, otherwise fetched from the Hub (header bytes only, no weight download).

**Memory**, per GPU, follows how vLLM sizes its KV cache at startup:

```
budget   = total VRAM × gpu-memory-utilization
KV cache = budget − weights − activation peak − CUDA graphs − runtime overhead
KV bytes per token = 2 × layers × kv_heads/tp × head_dim × kv dtype size
```

The estimator handles GQA, MLA (DeepSeek), sliding-window layers, MoE, FP8 and pre-quantized checkpoints, and tensor and pipeline parallelism. It also warns when `gpu-memory-utilization` asks for more than is currently free. This check matters on desktops and WSL, where the display already holds some VRAM.

**Throughput** uses a roofline model. Each decode step reads the weights plus the batch's KV cache from memory, or is limited by compute, whichever is slower, plus a fixed scheduling overhead. The effective batch is the number of average requests whose KV fits in the cache, capped at `max-num-seqs`. Treat uncalibrated results as an upper bound: the model leaves out per-step costs like kernel launches and the Python scheduler loop, which can dominate small models, eager mode and WSL. Run `vct bench` once per GPU and mode to calibrate them.

## Develop

```bash
uv run pytest            # tests/fake_vllm.py stands in for vLLM, so no GPU is needed
uv run ruff check src tests && uv run ruff format src tests
```
