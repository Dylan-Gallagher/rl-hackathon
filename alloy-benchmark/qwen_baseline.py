"""Untrained Qwen3-8B baseline over the SAME harness/tools/turn-cap as the alloy
experiment, to locate the difficulty band where an 8B has measurable signal.

Serving: Qwen3-8B is self-served with vLLM (`--enable-auto-tool-choice
--tool-call-parser hermes --reasoning-parser qwen3`, thinking disabled) rather
than a hosted API, because fine-tuned checkpoints won't exist on any API and a
serving-stack difference (chat template, sampling defaults, tokenizer, tool-call
parsing) would confound the comparison. The live model backend is wired into the
harness as condition "Q" (see llm_clients.call_qwen); it feeds from the same
neutral Conversation via the same to_openai / parse_openai_response adapters used
for GLM, so the baseline is directly comparable. Only the backend changes — each
challenge still runs in its own solver sandbox with the same run_command /
submit_flag tools and 40-turn cap.

This module holds the baseline-specific pieces: the format-failure classifier
(used by agent.py to log mechanical failures SEPARATELY from task failures) and
the difficulty-spread eval-set loaders:
- InterCode-CTF (100 picoCTF tasks) as the easy anchor (~46% published Pass@1).
- CTF-Dojo sha256+flagCheck challenges across categories as the harder band
  (Cybench/NYU-CTF territory, ~<5% published).
Report is per-challenge so mechanical (small-model) failure is distinguishable
from genuine capability failure.
"""
from __future__ import annotations
import json, re, time
from collections import Counter, defaultdict
from pathlib import Path

import config
from conversation import Conversation, parse_openai_response

# --- these MUST match agent.py exactly for cross-run comparability ----------
# (Inlined rather than imported so this module can't be perturbed by, or
# perturb, the parent's live edits to agent.py. Verify they are identical
# before the run.)
from agent import TOOLS, SYSTEM_PROMPT  # noqa: F401  (import is the comparability check)

QWEN_MODEL = "Qwen/Qwen3-8B"
VLLM_PORT = 8000

# Qwen3 non-thinking recommended sampling (Qwen team). Set explicitly so the
# baseline isn't at the mercy of a serving default — the exact confound the
# directive calls out.
QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.0)


# ===========================================================================
# 1. Serve Qwen3-8B with vLLM in a Daytona GPU sandbox
# ===========================================================================
def serve_qwen():
    """Boot an ephemeral H100 sandbox, start vLLM with tool-calling enabled and
    thinking disabled, and return (sandbox, openai_base_url, headers)."""
    from daytona import (Daytona, DaytonaConfig, CreateSandboxFromImageParams,
                         Resources, GpuType, SessionExecuteRequest)
    d = Daytona(DaytonaConfig(api_key=config.DAYTONA_API_KEY))
    sb = d.create(
        CreateSandboxFromImageParams(
            # vLLM/torch/CUDA are heavy; use a CUDA base and pip install, or a
            # prebuilt vllm image. 60 GiB holds the image + Qwen3-8B bf16 (~16 GB).
            image="vllm/vllm-openai:latest",
            resources=Resources(cpu=8, memory=32, disk=80, gpu=1, gpu_type=GpuType.H100),
            ephemeral=True,  # REQUIRED: GPU sandboxes must be ephemeral
        ),
        timeout=600,
    )
    sb.process.create_session("vllm")
    # --tool-call-parser hermes + --enable-auto-tool-choice: Qwen3 tool calling.
    # --reasoning-parser qwen3 lets us request thinking off per-call.
    serve_cmd = (
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"--model {QWEN_MODEL} --port {VLLM_PORT} "
        f"--enable-auto-tool-choice --tool-call-parser hermes "
        f"--reasoning-parser qwen3 --max-model-len 32768 "
        f"> /tmp/vllm.log 2>&1"
    )
    sb.process.execute_session_command("vllm", SessionExecuteRequest(command=serve_cmd, run_async=True))
    # wait for readiness
    for _ in range(120):
        r = sb.process.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:%d/v1/models" % VLLM_PORT, timeout=20)
        if (r.result or "").strip().endswith("200"):
            break
        time.sleep(10)
    link = sb.get_preview_link(VLLM_PORT)
    base_url = link.url.rstrip("/") + "/v1"
    headers = {"x-daytona-preview-token": link.token}
    return sb, base_url, headers


def make_qwen_caller(base_url: str, headers: dict):
    """Returns a call_qwen(conv, tools)->(parts, usage) with the SAME signature
    as llm_clients.call_glm, feeding from the neutral Conversation."""
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=base_url, default_headers=headers)

    def call_qwen(conv: Conversation, tools: list[dict]):
        messages = conv.to_openai()
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            max_tokens=4096,
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools],
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **QWEN_SAMPLING,
        )
        parts = parse_openai_response(resp)
        usage = {"input_tokens": resp.usage.prompt_tokens,
                 "output_tokens": resp.usage.completion_tokens,
                 "model": QWEN_MODEL,
                 "raw_message": resp.choices[0].message}
        return parts, usage

    return call_qwen


# ===========================================================================
# 2. Format-failure instrumentation (separate from task failure)
# ===========================================================================
def classify_format_failures(turn_records: list[dict]) -> dict:
    """Detect mechanical failure modes small models fall into, distinct from
    'tried and got the wrong answer'."""
    ff = Counter()
    last_calls = []
    for tr in turn_records:
        calls = tr.get("tool_calls", [])
        text = tr.get("assistant_text", "")
        # (a) no tool call at all this turn (model emitted only prose)
        if not calls:
            ff["no_tool_call"] += 1
        # (b) unparseable tool arguments (parser stored _raw_arguments)
        for c in calls:
            if isinstance(c.get("input"), dict) and "_raw_arguments" in c["input"]:
                ff["unparseable_tool_args"] += 1
        # (c) repetition loop: identical command to the immediately prior turn
        sig = json.dumps([(c["name"], c["input"]) for c in calls], sort_keys=True)
        if sig and last_calls and sig == last_calls[-1]:
            ff["repeated_identical_call"] += 1
        last_calls.append(sig)
        # (d) degenerate repetition inside the text (token loop)
        if text and _looks_repetitive(text):
            ff["text_repetition_loop"] += 1
    return dict(ff)


def _looks_repetitive(text: str, window: int = 40, thresh: int = 4) -> bool:
    toks = text.split()
    if len(toks) < window * 2:
        return False
    chunk = " ".join(toks[-window:])
    return text.count(chunk) >= thresh


# ===========================================================================
# 3. Difficulty-spread eval set
# ===========================================================================
def ctfdojo_spread(manifest_path: Path, per_category: int = 4) -> list[dict]:
    """A spread across CTF-Dojo categories (the harder band). Uses the existing
    sha256+flagCheck manifest so verification is identical to the alloy runs."""
    manifest = json.load(open(manifest_path))
    by_cat = defaultdict(list)
    for k, v in manifest.items():
        by_cat[v["category"]].append(v)
    out = []
    for cat, items in by_cat.items():
        out.extend(items[:per_category])
    return out


def load_intercode_ctf(cache: Path) -> list[dict]:
    """The easy anchor: 100 picoCTF tasks with plaintext gold flags (direct
    string-match verification). Downloads ic_ctf.json + task_assets on first use.
    Returns normalized challenge dicts."""
    import urllib.request
    url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ctf/ic_ctf.json"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "curl"}), timeout=30).read())
    tasks = json.loads(cache.read_text())
    return [{
        "id": f"ic-ctf-{t['task_id']}",
        "category": (t.get("tags") or ["misc"])[0],
        "query": t["query"],
        "gold_flag": t["gold"],           # plaintext -> direct match
        "verify": "plaintext",
        "source": "intercode-ctf",
    } for t in tasks]


# ===========================================================================
# 4. Entry point (blocked until GPU credits exist)
# ===========================================================================
if __name__ == "__main__":
    print(__doc__)
    print("\nEval-set composition this module would run:")
    spread = ctfdojo_spread(config.CHALLENGES_DIR / "manifest.json", per_category=4)
    print(f"  CTF-Dojo (hard band): {len(spread)} challenges across categories")
    ic = load_intercode_ctf(Path("/tmp/ic_ctf.json"))
    tags = Counter(t["category"] for t in ic)
    print(f"  InterCode-CTF (easy anchor): {len(ic)} tasks, tags={dict(tags)}")
    print("\nRun blocked: add GPU credits to the Daytona org wallet, then call "
          "serve_qwen() + run the standard agent loop with make_qwen_caller().")
