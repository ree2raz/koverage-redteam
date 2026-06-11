"""Serve OpenAI's gpt-oss models on Modal via vLLM (OpenAI-compatible /v1 API).

This is the red-team TARGET endpoint. We self-host the open weights rather than
calling a hosted gpt-oss endpoint on purpose: hosting the weights removes the
model-provider-ToS red-teaming prohibition (only the Apache-2.0 license applies),
gives a *bare* model with no provider-injected moderation wrapper, and pins the
exact weights/serving config for reproducibility.

Ladder (one-line change): set MODEL_KEY to "20b" to start, "120b" to move up.
The 20b model fits in ~16GB; 120b needs an 80GB GPU (H100/A100-80GB).

Deploy:
    modal deploy deploy/modal_gpt_oss.py

Modal prints a URL like https://<you>--redteam-gpt-oss-serve.modal.run
Put it in .env as MODAL_OSS_URL (the harness appends /v1 itself).

Tool calling is enabled with the gpt-oss-specific parser
(`--tool-call-parser openai --enable-auto-tool-choice`) so the receptionist
agent's function calls work over /v1/chat/completions.

Based on Modal's official example: https://modal.com/docs/examples/gpt_oss_inference
"""

from __future__ import annotations

import json

import modal

# ---------------------------------------------------------------------------
# Ladder config — change MODEL_KEY to climb from 20b to 120b.
# ---------------------------------------------------------------------------

MODEL_KEY = "20b"

# Pinned revisions for reproducibility. Update via `huggingface-cli` if you
# deliberately bump weights — and note the bump in a transcript/run record.
MODELS = {
    "20b": {
        "name": "openai/gpt-oss-20b",
        "revision": "d666cf3b67006cf8227666739edf25164aaffdeb",
        # gpt-oss is MXFP4-NATIVE. On Ampere (A100) vLLM has no MXFP4 kernel, so it
        # dequantizes to bf16 (slow) AND we must --enforce-eager (no CUDA graphs):
        # both worst-case penalties at once (~7 tok/s observed). H100 (Hopper) has
        # the native Triton matmul_ogs MXFP4 kernel + CUDA graphs + fp8 KV cache —
        # a ~3-5x latency win. Per-second cost is higher but the run finishes
        # proportionally sooner, so a fixed workload is roughly cost-neutral.
        # 20b (~16GB) leaves ample headroom on an 80GB H100.
        "gpu": "H100:1",
    },
    "120b": {
        "name": "openai/gpt-oss-120b",
        # Pin this to a concrete commit before any scored 120b run.
        "revision": None,
        "gpu": "H100:1",
    },
}

CFG = MODELS[MODEL_KEY]
MODEL_NAME = CFG["name"]
MODEL_REVISION = CFG["revision"]
GPU = CFG["gpu"]

# fp8 KV-cache (fp8e4nv / E4M3) and the FlashInfer MXFP4 MoE kernels only exist
# on Hopper+ (H100/H200/B200). On Ampere (A100) they fail to compile
# ("type fp8e4nv not supported in this architecture"), so we gate them. gpt-oss
# still runs on A100 via vLLM's Triton MXFP4 dequant path — just without these
# throughput extras.
_GPU_KIND = GPU.split(":")[0].split("-")[0].upper()
_IS_HOPPER_PLUS = _GPU_KIND in {"H100", "H200", "B200"}

VLLM_PORT = 8000
VLLM_VERSION = "0.18.1"

# Keep one container warm for 10 min after the last request so a batch campaign
# amortizes the cold start; scales to zero afterward to save credits.
SCALEDOWN_WINDOW = 600
STARTUP_TIMEOUT = 1800

# ---------------------------------------------------------------------------
# Image + caches
# ---------------------------------------------------------------------------

_IMAGE_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
if _IS_HOPPER_PLUS:
    # Faster MXFP4 MoE kernels — Hopper/Blackwell only.
    _IMAGE_ENV["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "1"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]==0.36.0",
    )
    .env(_IMAGE_ENV)
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("redteam-gpt-oss")

# vLLM server flags. Tool calling uses the gpt-oss `openai` parser — using any
# other parser (hermes/llama3_json/mistral) fails or mangles arguments.
# NOTE: automatic prefix caching is ON by default in vLLM v1, which reuses the KV
# cache for the shared system prompt + growing history across the many turns of a
# multi-turn attack — a large prefill saving for our workload, no flag needed.
VLLM_CONFIG = {
    "max-model-len": 32768,
    # 8192 is the GPT-OSS recipe's recommended batched-token size (avoids the
    # OOM the recipe warns about at 16384 on a single H100, TP=1).
    "max-num-batched-tokens": 8192,
}
if _IS_HOPPER_PLUS:
    # fp8 KV cache needs E4M3 (fp8e4nv) — Hopper+ only; omit on Ampere.
    VLLM_CONFIG["kv-cache-dtype"] = "fp8"
    # Cap CUDA-graph capture so we keep the graph speedup without over-spending
    # GPU memory on capture (vLLM's recommended value).
    VLLM_CONFIG["max-cudagraph-capture-size"] = 2048


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # --- tool / function calling for gpt-oss ---
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        # --- serving knobs ---
        "--async-scheduling",
        "--tensor-parallel-size",
        "1",
    ]
    # CUDA graph compilation (torch.compile) is reliable on Hopper+; on Ampere
    # it can hit Triton/inductor errors with gpt-oss's MXFP4 path, so fall back
    # to eager there (slower, but bulletproof for the smoke run).
    cmd.append("--no-enforce-eager" if _IS_HOPPER_PLUS else "--enforce-eager")
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    cmd += [item for k, v in VLLM_CONFIG.items() for item in (f"--{k}", str(v))]

    print("Launching:", json.dumps(cmd))
    subprocess.Popen(cmd)
