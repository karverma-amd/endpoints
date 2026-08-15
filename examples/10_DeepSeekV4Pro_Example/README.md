# DeepSeek-V4-Pro Benchmark (SGLang)

End-to-end example for benchmarking `deepseek-ai/DeepSeek-V4-Pro` with SGLang, using the same
performance and accuracy datasets as the [GPT-OSS-120B example](../04_GPTOSS120B_Example/Readme.md)
(AIME25, GPQA, LiveCodeBench).

The server uses `/v1/chat/completions` (`api_type: openai`) with text prompts from the dataset.
The server applies the DeepSeek-V4 chat template and reasoning parser.

## Getting the Dataset

The performance dataset must be obtained from the LLM task-force (parquet format). Place it at:

```
examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet
```

The accuracy datasets (AIME25, GPQA, LiveCodeBench) are downloaded automatically from HuggingFace.

## Environment Setup

```bash
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
export HF_TOKEN=<your HuggingFace token>  # required for GPQA (gated) and faster HF downloads
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
export MODEL_DIR=/data/workloads-inference/models
export MODEL_PATH=${MODEL_DIR}/deepseek-ai/DeepSeek-V4-Pro
export TOKENIZER_MODEL_PATH=${MODEL_PATH}  # host path for ISL/OSL/TPOT metrics
```

Preflight scripts (`run_sglang_benchmark.sh`, `run_sglang_accuracy_benchmark.sh`) probe the inference server with `GET /health` and `GET /v1/models`. Override the base URL or wait time while a server is starting:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000    # default when SGLANG_PORT=30000
export WAIT_FOR_SGLANG_S=120                      # seconds; 0 = single attempt
```

## Download Model

Download weights to the shared model store and mount them into the serving container:

```bash
mkdir -p "${MODEL_PATH}"
hf download "${MODEL_NAME}" --local-dir "${MODEL_PATH}"
```

---

## SGLang (ROCm / MI35x)

SGLang serves DeepSeek-V4-Pro on ROCm using the DSv4 `_prs` image
(`rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs`), which bakes open
sglang/aiter PRs needed for shared-experts fusion, aiter mHC, and bpreshuffle
no-copy scale paths. Before launch, patch `config.json` so `model_type` is
`deepseek_v3` (SGLang registry compatibility).

### Launch Server

**Option A: helper script (host or container)**

On a ROCm host with SGLang installed (must be the `_prs` build, or set
`VERIFY_BAKED_PATCHES=false`):

```bash
export MODEL_PATH=/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro
export SGLANG_PORT=30000
export TP=8
export CONC=512
./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
```

Via Docker (model directory must exist on the host):

```bash
export MODEL_PATH=/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro
export RUN_MODE=docker
export SGLANG_IMAGE=rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs
./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
```

Optional overrides:

| Variable                    | Default                                                         | Description                                                                                                                   |
| --------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_PORT` / `HTTP_PORT` | `30000`                                                         | HTTP listen port (`start_sglang_server.sh` unsets `SGLANG_PORT` before launch — SGLang uses that name for internal ZMQ ports) |
| `TP`                        | `8`                                                             | Tensor parallel size                                                                                                          |
| `CONC`                      | `512`                                                           | `--max-running-requests` / `--cuda-graph-max-bs`                                                                              |
| `ISL`                       | `8192`                                                          | Chunked-prefill size (`ISL * TP` when `DP_ATTENTION=true`)                                                                    |
| `MAX_MODEL_LEN`             | `327680`                                                        | `--context-length`                                                                                                            |
| `DP_ATTENTION`              | `false`                                                         | `true` enables DP attention + disables shared-experts fusion; `false` uses `--enforce-shared-experts-fusion`                  |
| `EP_SIZE`                   | `1`                                                             | Expert parallel size (`>1` adds `--ep-size`)                                                                                  |
| `CHAT_TEMPLATE`             | `chat_templates/deepseek_v4_thinking.jinja`                     | Thinking chat template passed to `--chat-template`                                                                            |
| `SGLANG_IMAGE`              | `rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs`      | Docker image when `RUN_MODE=docker`                                                                                           |
| `VERIFY_BAKED_PATCHES`      | `true`                                                          | Refuse to launch if the baked sglang/aiter PR markers are missing                                                             |

The script exports `SGLANG_USE_AITER=1`, `SGLANG_OPT_*AITER_BATCHED_GEMM=1`, and
the DSv4 thinking/effort flags, then launches (via `sglang serve` when available):

```text
sglang serve \
  --model-path $MODEL \
  --tensor-parallel-size $TP \
  --attention-backend dsv4 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --chat-template .../deepseek_v4_thinking.jinja \
  ...
```

**Option B: manual launch** (same flags as `start_sglang_server.sh`)

```bash
# Patch HF config (once per cache checkout)
python3 <<'PYEOF'
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="deepseek-ai/DeepSeek-V4-Pro", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
PYEOF

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_USE_AITER=1
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_WO_A_AITER_BATCHED_GEMM=1
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1

sglang serve \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 30000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend dsv4 \
  --cuda-graph-max-bs 512 \
  --max-running-requests 512 \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.15 \
  --page-size 256 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 327680 \
  --chunked-prefill-size 8192 \
  --enforce-shared-experts-fusion \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --chat-template examples/10_DeepSeekV4Pro_Example/chat_templates/deepseek_v4_thinking.jinja \
  --watchdog-timeout 1800
```

Verify:

```bash
curl http://127.0.0.1:30000/health
```

### Run Benchmark (SGLang)

[`sglang_deepseek_v4_pro_example.yaml`](sglang_deepseek_v4_pro_example.yaml) targets
`http://localhost:30000` with the performance + AIME25 + GPQA + LiveCodeBench datasets:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_example.yaml \
  --timeout 60
```

Or use the helper script:

```bash
./examples/10_DeepSeekV4Pro_Example/run_sglang_benchmark.sh
```

Performance-only:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_perf.yaml \
  --timeout 60
```

### Run Accuracy (SGLang)

Same workflow as [GPT-OSS `run.py`](../04_GPTOSS120B_Example/Readme.md#accuracy-suite-script):
start SGLang, start `lcb-service`, set `HF_TOKEN` (required for gated GPQA), then run the
accuracy helper (checks all prerequisites, tees logs under `results/docker_logs/accuracy/`):

```bash
export HF_TOKEN=<your HuggingFace token>
export HF_HOME=~/.cache/huggingface

./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
./examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh
./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

YAML-only (equivalent):

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_accuracy.yaml \
  --timeout 3600
```

| Argument / env          | Default      | Description                                  |
| ----------------------- | ------------ | -------------------------------------------- |
| `HF_TOKEN`              | _(required)_ | HuggingFace token for GPQA download          |
| `TIMEOUT`               | `86400`      | Benchmark timeout (seconds)                  |
| `DOCKER_LOG_STORAGE_GB` | `64`         | Container writable layer size when supported |

Accuracy config uses `max_new_tokens: 320000`, `num_workers: 32`, `num_repeats: 4` per dataset,
and phase order AIME25 → GPQA → LiveCodeBench.

### Docker log storage

Server and LCB containers mount `results/docker_logs/<service>/` on the host at `/workspace`.
Scripts also pass `--storage-opt size=16G` on `docker run` when the daemon uses overlay2
(override with `DOCKER_LOG_STORAGE_GB`). SGLang server stdout is written to
`results/docker_logs/sglang/server.log`.

### Config notes

- **Performance dataset**: `text_input` → `prompt`. Server applies DeepSeek-V4 chat template.
- **Accuracy datasets**: `::deepseek_v4` presets (same prompt formatting as GPT-OSS).
- **Reasoning output**: server streams reasoning separately; client accumulates `reasoning_content`
  and final `content` for scoring.
- **`model_params.name`**: use the HuggingFace id (`deepseek-ai/DeepSeek-V4-Pro`). Set
  `TOKENIZER_MODEL_PATH` to the host weights path for ISL/OSL/TPOT.

---

## LiveCodeBench Setup

LiveCodeBench has dependency conflicts with the main package. Two options:

### Option A: containerized scorer (recommended when `dhi.io` access is available)

Follow the [LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container).
Requires `docker login dhi.io`, then:

```bash
./examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh
curl http://127.0.0.1:13835/info
```

### Option B: local subprocess scorer (no `docker login dhi.io`)

Same fallback as the GPT-OSS example — runs `lcb_serve` as a subprocess on the host:

```bash
export ALLOW_LCB_LOCAL_EVAL=true
./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

The accuracy helper skips the `:13835` preflight when this is set (default `true`).
WebSocket scoring is attempted first if `lcb-service` is up; otherwise scoring falls
back to the subprocess path automatically.

---

## Troubleshooting

**Cannot connect to SGLang server**

- Verify: `curl http://localhost:30000/health`
- Confirm `SGLANG_PORT` matches `endpoint_config.endpoints` in the SGLang YAML
- For ROCm, use the DSv4 `_prs` image:
  `rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs`

**SGLang `address already in use` on `--port`**

- Do not export `SGLANG_PORT` into the SGLang process (upstream reserves it for ZMQ).
  Use `start_sglang_server.sh`, which passes `--port` on the CLI and runs with
  `env -u SGLANG_PORT`.

**SGLang fails to load model / unknown architecture**

- Run the `config.json` patch (`model_type`: `deepseek_v4` → `deepseek_v3`) before launch
- Confirm `SGLANG_USE_AITER=1` and that baked-patch verification passes on the `_prs` image

**LiveCodeBench scoring fails / Connection refused on port 13835**

- Start `lcb-service` (`docker login dhi.io` + `start_lcb_service.sh`), or
- `export ALLOW_LCB_LOCAL_EVAL=true` and re-run (see LiveCodeBench Setup above)

**Out of memory**

- Increase `TP` / `--tensor-parallel-size`
- Lower `--mem-fraction-static`

**Model not found in container**

- Confirm the host path exists: `ls "${MODEL_PATH}"`
- Mount into the container at the path used by `--model-path`

**docker build fails with `dhi.io ... unauthorized`**

- Run `docker login dhi.io` with your Docker Hub credentials (PAT with read access to hardened images)
