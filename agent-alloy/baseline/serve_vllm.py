"""Serve Qwen3-8B (or a fine-tuned variant) with vLLM in a Daytona GPU sandbox.

Self-hosting (not a hosted API) is deliberate: later phases use fine-tuned
variants that exist on no API, and we must hold the serving stack — chat
template, sampling defaults, tokenizer handling, tool-call parser — FIXED so it
doesn't confound comparisons against this baseline. Every such knob is pinned
explicitly below rather than left to a provider default.

Returns a public OpenAI-compatible base_url that alloy.providers.OpenAICompatAdapter
can point at, so the baseline runs through the exact same agent harness.

NOTE: requires Daytona GPU credits (org currently has none -> create fails with
"Organization doesn't have GPU credits"). Untested end-to-end for that reason;
the serving recipe follows vLLM + Qwen3 docs.
"""
from __future__ import annotations
import os, time, json
from dotenv import load_dotenv
from daytona import (Daytona, DaytonaConfig, CreateSandboxFromImageParams,
                     Resources, GpuType)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen3-8B")
PORT = 8000
# Pinned sampling (Qwen3 non-thinking eval preset) — recorded so it's reproducible
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "repetition_penalty": 1.05, "max_tokens": 4096}


def _vllm_cmd() -> str:
    # --enable-auto-tool-choice + hermes parser: Qwen3 tool-calling format.
    # --chat-template left to the tokenizer's built-in (override via env for
    # fine-tuned variants). --tokenizer-mode auto, trust remote code.
    tmpl = os.environ.get("BASELINE_CHAT_TEMPLATE", "")
    tmpl_arg = f" --chat-template {tmpl}" if tmpl else ""
    return (
        "python3 -m vllm.entrypoints.openai.api_server "
        f"--model {MODEL} --served-model-name {MODEL} "
        f"--host 0.0.0.0 --port {PORT} "
        "--enable-auto-tool-choice --tool-call-parser hermes "
        "--reasoning-parser qwen3 "          # split <think> out of content
        "--max-model-len 32768 --gpu-memory-utilization 0.92 "
        "--dtype bfloat16 --trust-remote-code" + tmpl_arg)


def start(gpu: GpuType = GpuType.RTX_4090, ttl_minutes: int = 90) -> dict:
    d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    img = (CreateSandboxFromImageParams(
        image="vllm/vllm-openai:latest",     # ships CUDA + vLLM
        resources=Resources(cpu=8, memory=32, disk=60, gpu=1, gpu_type=gpu),
        ephemeral=True))                     # GPU sandboxes must be ephemeral
    print(f"creating GPU sandbox ({gpu}) for {MODEL} ...")
    sb = d.create(img, timeout=600)
    print("sandbox", sb.id, "- launching vLLM (model download + load)...")
    # launch server in a background session
    sess = "vllm"
    sb.process.create_session(sess)
    sb.process.execute_session_command(
        sess, __import__("daytona").SessionExecuteRequest(
            command=f"nohup {_vllm_cmd()} > /tmp/vllm.log 2>&1 &", var_async=True))
    # wait for health
    base = sb.get_preview_link(PORT).url.rstrip("/")
    print("waiting for vLLM /health at", base)
    import requests
    for i in range(120):
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200:
                print(f"vLLM up after ~{i*10}s"); break
        except Exception:
            pass
        time.sleep(10)
    else:
        log = sb.process.exec("tail -40 /tmp/vllm.log", timeout=30).result
        raise RuntimeError("vLLM did not become healthy:\n" + (log or ""))
    info = {"sandbox_id": sb.id, "base_url": base + "/v1", "model": MODEL,
            "sampling": SAMPLING, "cmd": _vllm_cmd()}
    json.dump(info, open(os.path.join(os.path.dirname(__file__),
                                      "results", "serving.json"), "w"), indent=2)
    print("serving:", json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    start()
