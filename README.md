# Qwen 27B vLLM Docker

Docker-based vLLM serving for the latest Qwen 27B model, currently [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with [Frozenlock AutoRound INT4 quant](https://huggingface.co/Frozenlock/Qwen3.8-27B-int4-AutoRound) and MTP speculative decoding. This repo always tracks the newest Qwen 27B release — when a new version drops, the `latest` tag moves forward. Older versions remain available as version-tagged images in the registry (e.g. `:v3.6`, `:v3.8`). The [Qwen3.6-27B image](https://github.com/tedivm/qwen36-27b-docker/pkgs/container/qwen36-27b-docker) is retained in the old registry. Model is downloaded at runtime and stored on a host volume so the container can be upgraded without redownloading weights.

## What you get

| Metric (dual RTX 3090, TP=2, 250K context) | Value |
|---|---|
| Sustained TPS on coding workloads | **~121** |
| Sustained TPS on prose | ~87 |
| Max context length | 250,000 tokens (native: 262K) |
| Vision support | yes (MoonViT, via `image_url` content parts) |

The KV cache token size can hit a total of ~544K tokens. You can have one concurrent session with a max context length of 544K, two concurrent sessions with a max context length of 272K.

I typically run with three concurrent sequences allowing for a max context length of 200K per request, but if all three sessions take advantage of that then I will get an OOM error. In practice this has yet to occur.

### Verified benchmarks (2x RTX 3090, TP=2)

Measured with `bench_tps.py` against the Docker container. Tuned with `--performance-mode interactivity --gdn-prefill-backend triton --kda-decode-backend flashinfer` (KDA prefill on `auto`):

| Workload | Tokens | Time | TPS | MTP Accept |
|---|---|---|---|---|
| Prose (800-word story) | 800 | 9.25s | **86.51** | 39.8% |
| Code (LRU cache impl) | 1200 | 9.90s | **121.26** | 69.6% |

**Power-limited (250W per GPU):**

| Workload | Tokens | Time | TPS | MTP Accept |
|---|---|---|---|---|
| Prose (800-word story) | 800 | 10.02s | **79.80** | 35.6% |
| Code (LRU cache impl) | 1200 | 12.76s | **94.01** | 48.0% |

**Qwen3.6-27B reference (same hardware, 200K context, vLLM 0.23):**

| Workload | Tokens | Time | TPS |
|---|---|---|---|
| Prose (800-word story) | 800 | 8.38s | **95.00** |
| Code (LRU cache impl) | 1200 | 9.68s | **124.00** |

**Qwen3.6-27B power-limited (250W per GPU):**

| Workload | Tokens | Time | TPS |
|---|---|---|---|
| Prose (800-word story) | 800 | 8.64s | **92.56** |
| Code (LRU cache impl) | 1200 | 10.86s | **110.47** |

## Hardware

Tested on 2x RTX 3090 (48 GB VRAM total). Also works at lower context on a single 24 GB card. GPU count is auto-detected — pass `--gpus all` for multi-GPU or `--gpus '"device=0"'` for single-GPU.

Minimum disk: ~22 GB for weights + ~6 GB for caches.

## Quick start

### Option 1: Docker Compose (recommended)

Create a `docker-compose.yml`:

```yaml
services:
  qwen27b:
    image: ghcr.io/tedivm/qwen-27b-docker:latest
    container_name: qwen27b
    runtime: nvidia
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - MODEL_DOWNLOAD=1
    shm_size: 1g
    volumes:
      - ./models:/data/models
      - ./logs:/data/logs
    ports:
      - "1234:1234"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then:

```bash
# Download model (one-time, stores in ./models)
docker compose run --rm qwen27b download

# Start the server
docker compose up -d

# Stop
docker compose down
```

### Option 2: Docker CLI

```bash
# Pull from GHCR
docker pull ghcr.io/tedivm/qwen-27b-docker:latest

# Download model (one-time, stores in /path/on/host/models)
docker run --rm --gpus all \
  -v /path/on/host/models:/data/models \
  ghcr.io/tedivm/qwen-27b-docker:latest download

# Start server (auto-detects 1 or 2 GPUs)
docker run -d --name qwen27b --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  ghcr.io/tedivm/qwen-27b-docker:latest

# Single-GPU override
docker run -d --name qwen27b --gpus '"device=0"' -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  ghcr.io/tedivm/qwen-27b-docker:latest

# Upgrade (no redownload needed)
docker stop qwen27b && docker rm qwen27b
docker pull ghcr.io/tedivm/qwen-27b-docker:latest
docker run -d --name qwen27b --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  ghcr.io/tedivm/qwen-27b-docker:latest
```

## Accessing the Server

The server exposes an OpenAI-compatible API on port 1234.

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Authorization: Bearer any" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen_27b","messages":[{"role":"user","content":"hi"}]}'
```

### Version-agnostic model name

The server responds to both `qwen3.8-27b` (the current version) and `qwen_27b` (a stable alias). Use `qwen_27b` in your client config so you can upgrade the Docker image to a new Qwen release without changing any client-side settings.

### OpenCode

Add a provider to your `opencode.jsonc`:

```jsonc
{
  "providers": {
    "vllm": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://localhost:1234/v1",
        "chunkTimeout": 18000000,
      },
      "models": {
        "qwen_27b": {
          "name": "Qwen 27B",
          "capabilities": {
            "tools": true,
            "input": ["text", "image"],
            "output": ["text"],
          },
          "limit": {
            "context": 262144,
            "output": 20000,
          },
        },
      },
    },
  },
  "model": "vllm/qwen_27b",
}
```

> **Note:** OpenCode v1 has a [tool-call parsing bug](https://github.com/anomalyco/opencode/issues/26412) with the `openai-compatible` package — use the `anthropic` package instead if you're on v1. Fixed in v2.

## GPU-specific tuning

Qwen3.8 uses a hybrid attention architecture: each block of 4 layers contains 3 GDN (Gated Delta Net, linear attention) layers and 1 KDA (full attention) layer. vLLM 0.30+ exposes backend selection for each:

| Flag | Controls | Options |
|---|---|---|
| `GDN_PREFILL_BACKEND` | GDN prefill (75% of layers) | `flashinfer`, `triton`, `cutedsl` |
| `KDA_PREFILL_BACKEND` | KDA prefill (25% of layers) | `auto`, `triton`, `flashkda`, `flashinfer`, `fused` |
| `KDA_DECODE_BACKEND` | KDA decode (runs every token) | `auto`, `native`, `flashinfer`, `triton` |
| `PERFORMANCE_MODE` | CUDA graph / kernel tuning | `balanced`, `interactivity`, `throughput` |

**Recommended settings by GPU:**

| GPU | Architecture | GDN Prefill | KDA Prefill | KDA Decode | Performance Mode |
|---|---|---|---|---|---|
| RTX 3090 | Ampere (SM86) | `triton` | `auto` | `flashinfer` | `interactivity` |
| RTX 4090 | Ada (SM89) | `triton` | `auto` | `flashinfer` | `interactivity` |
| RTX 5090 | Blackwell (SM120) | `cutedsl` | `auto` | `flashinfer` | `interactivity` |
| A100 / H100 | Datacenter | `triton` / `cutedsl`* | `auto` | `flashinfer` | `throughput` |

\* H100 (SM90) supports `cutedsl`; A100 (SM80) should use `triton`.

**Why these settings:**

- **`triton` for GDN prefill** — Triton kernels are well-optimized for Ampere/Ada. `cutedsl` is only available on Hopper (SM90+) and Blackwell (SM120+).
- **`auto` for KDA prefill** — vLLM's automatic selection picks the best available kernel for your hardware. Explicit backends like `flashkda` can be slower than the auto-selected path.
- **`flashinfer` for KDA decode** — FlashInfer's decode kernels are the fastest for single-token generation. This runs on every generated token, so it has the biggest impact on sustained TPS.
- **`interactivity` mode** — Favors low per-request latency with fine-grained CUDA graphs. Best for single-user workstations. Use `throughput` for multi-tenant serving.

**Overriding settings:**

```bash
# Example: try different backends
docker compose down
GDN_PREFILL_BACKEND=flashinfer KDA_DECODE_BACKEND=triton docker compose up -d

# Or set in .env
echo 'GDN_PREFILL_BACKEND=flashinfer' >> .env
```

## Environment variables

All configuration is via environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/data/models` | Model weights path (mount from host) |
| `MODEL_REPO` | `Frozenlock/Qwen3.8-27B-int4-AutoRound` | HuggingFace model repo |
| `PORT` | `1234` | API port |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Model name for API |
| `MAX_MODEL_LEN` | `250000` | Max context length (auto-lowered to 48000 for single GPU) |
| `MAX_NUM_SEQS` | `3` | Concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max tokens batched per step (auto-lowered to 4096 for single GPU) |
| `GPU_MEMORY_UTIL` | `0.92` | GPU memory fraction (auto-set to 0.95 for single GPU) |
| `NUM_SPECULATIVE_TOKENS` | `3` | MTP speculative decoding tokens |
| `KV_CACHE_DTYPE` | `fp8` | KV cache precision |
| `TENSOR_PARALLEL` | *(auto)* | Tensor parallelism (auto-detected from GPU count) |
| `TEMPERATURE` | `0.6` | Generation temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `TOP_K` | `20` | Top-k sampling |
| `MIN_P` | `0.0` | Min-p sampling threshold |
| `PRESENCE_PENALTY` | `0` | Presence penalty |
| `REPETITION_PENALTY` | `1.0` | Repetition penalty |
| `REASONING_PARSER` | `qwen3` | Reasoning parser (blank to disable) |
| `CHAT_TEMPLATE` | *(empty)* | Chat template (blank uses model default; set to `/usr/local/bin/qwen3.8-froggeric.jinja` for the enhanced template) |
| `CHAT_TEMPLATE_PRESERVE_THINKING` | `true` | Preserve thinking tags in chat template kwargs |
| `CHAT_TEMPLATE_ENABLE_THINKING` | `true` | Enable thinking in chat template kwargs |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | `0` | Opt out of FlashInfer all-reduce (recommended for NVLink-less boxes) |
| `PER_REQUEST_SPEC_DECODE` | `none` | Per-request MTP metrics: `none`, `summary`, or `detailed` |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode: `balanced`, `interactivity`, or `throughput` |
| `GDN_PREFILL_BACKEND` | `triton` | GDN prefill kernel: `flashinfer`, `triton`, or `cutedsl` (see GPU tuning) |
| `KDA_PREFILL_BACKEND` | *(empty/auto)* | KDA prefill kernel: `auto`, `triton`, `flashkda`, `flashinfer`, or `fused` |
| `KDA_DECODE_BACKEND` | `flashinfer` | KDA decode kernel: `auto`, `native`, `flashinfer`, or `triton` |
| `HF_TOKEN` | *(empty)* | HuggingFace auth token for gated models |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | *(empty)* | OTel traces endpoint (e.g. `grpc://otel-collector:4317`) |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | *(empty)* | OTel metrics endpoint (e.g. `grpc://otel-collector:4317`) |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | *(empty)* | OTel logs endpoint (e.g. `grpc://otel-collector:4317`) |
| `OTEL_EXPORTER_OTLP_INSECURE` | *(empty)* | Set to `true` for plaintext gRPC (OTel SDK env, applies to all signals) |
| `OTEL_EXPORTER_OTLP_TRACES_INSECURE` | *(empty)* | Traces-only plaintext override (OTel SDK env) |
| `OTEL_EXPORTER_OTLP_METRICS_INSECURE` | *(empty)* | Metrics-only plaintext override (OTel SDK env) |
| `OTEL_EXPORTER_OTLP_LOGS_INSECURE` | *(empty)* | Logs-only plaintext override (OTel SDK env) |

## Usage

**Monitor** (inside the container):

```bash
docker exec -it qwen38 watch-vllm.py
docker exec -it qwen38 watch-vllm.py /data/models/.tmp/vllm.log 24
```

**Benchmark** (inside the container):

```bash
docker exec -it qwen38 python bench_tps.py
```

**OpenAI-compatible API**:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Authorization: Bearer any" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}'
```

## Server flags

| Flag | Why |
|---|---|
| `--dtype auto` | Lets vLLM detect the model's native bfloat16 dtype |
| `--quantization auto_round` | Matches the Frozenlock weights |
| `--kv-cache-dtype fp8` | Halves KV memory vs FP16; 250K x 3 seqs fits on 48 GB (set via `KV_CACHE_DTYPE`) |
| `--enable-prefix-caching` | Not default for Qwen3.8 hybrid attention; opt in |
| `--enable-chunked-prefill` | Recommended alongside spec-decode for throughput |
| `--speculative-config method=mtp, num_speculative_tokens=3` | ~2x throughput on code; 3 is the sweet spot (set via `NUM_SPECULATIVE_TOKENS`) |
| `--max-num-seqs 3` | Solo user + subagents; raise for more concurrency |
| `--max-num-batched-tokens 8192` | Set via `MAX_NUM_BATCHED_TOKENS`; tune for throughput vs latency |
| `--gpu-memory-utilization 0.92` | Leaves CUDA-graph margin |
| `--disable-custom-all-reduce` | No NVLink — stock NCCL is faster |
| `--reasoning-parser qwen3` | Enables extended thinking output |
| `--chat-template` | Optional — set via `CHAT_TEMPLATE` to override model default (e.g. `qwen3.8-froggeric.jinja`) |
| `--default-chat-template-kwargs` | Set `preserve_thinking` and `enable_thinking` via `CHAT_TEMPLATE_PRESERVE_THINKING` and `CHAT_TEMPLATE_ENABLE_THINKING` |
| `--per-request-spec-decode-metrics` | MTP acceptance stats in API responses (set via `PER_REQUEST_SPEC_DECODE`) |
| `--performance-mode interactivity` | Fine-grained CUDA graphs, latency-oriented kernels (set via `PERFORMANCE_MODE`) |
| `--gdn-prefill-backend triton` | GDN prefill kernel selection (set via `GDN_PREFILL_BACKEND`) |
| `--kda-prefill-backend` | KDA prefill kernel selection (set via `KDA_PREFILL_BACKEND`, defaults to `auto`) |
| `--kda-decode-backend flashinfer` | KDA decode kernel selection (set via `KDA_DECODE_BACKEND`) |

TP=2 beats TP=1 by ~1.5x on dual 3090s. Memory-bandwidth savings from splitting weights across two cards outweigh the PCIe NCCL all-reduce cost.

## Tool calling

Tool calling uses `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` (controlled by `TOOL_CALL_PARSER` and `AUTO_TOOL_CHOICE` env vars). The froggeric chat template handles tool call formatting natively in both XML and JSON formats.

Simply pass `tools` in your API request with `tool_choice: "auto"` and the model handles the rest.

## Caveats

- **opencode v1 users: set your provider to `anthropic`** instead of `openai` due to [an opencode bug](https://github.com/anomalyco/opencode/issues/26412) that breaks tool call parsing. Fixed in opencode v2; v1 is still affected.
- **Mamba prefix caching is experimental** for Qwen3.8. vLLM auto-picks the `align` fallback mode for `Qwen3_5ForConditionalGeneration`. Regular-attention layers cache fine (~85% hit rate); Mamba/GDN linear-attention layers re-run prefill on every new request.
- **Spec-decode silently ignores** `min_p` and `logit_bias` per-request params.
- **Deprecation warnings about `Qwen2VLImageProcessorFast` / `use_fast`** are upstream-transformers noise; ignore.
- **CUDA graph mode downgrades** to `PIECEWISE` under spec-decode (FlashInfer limitation) — automatic and expected.

## Acknowledgments

- [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) — the original Qwen3.6 repo this was forked from; the vLLM flag tuning, performance benchmarks, and serve scripts all originated there and were adapted for Qwen3.8.
- [Frozenlock](https://huggingface.co/Frozenlock) — AutoRound INT4 quant using the same recipe as Lorbus (W4A16, GDN in_proj kept BF16, vision untouched, MTP quantized).
- [Qwen team](https://github.com/QwenLM/Qwen3) — the base model and the MTP head.
- Medium article ["An Overnight Stack for Qwen3.6-27B"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914?postPublishedType=repub) — original source of the AutoRound + MTP + TurboQuant stack.
- [Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) — more aggressive approach with TurboQuant KV; useful reference for pushing further.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — enhanced universal chat template (v22.5) covering Qwen 3.5, 3.6, and 3.8 with improved tool calling, reasoning effort control, and multi-step tool error handling.
