# Alloy Agents — Replication

A faithful replication of the technique in XBOW's
[*Agents Built From Alloys*](https://xbow.com/blog/alloy-agents): running a single
agent loop over **one shared conversation** while **randomly alternating which LLM
generates each step**. Each model sees the other's prior turns (text *and* tool
calls) as its own, unaware a swap happened. Total model-call count is identical to
a single-model run — only *who answers* each step varies.

The paper's headline finding: an alloy of two diverse (cross-provider) models
solves substantially more CTF/pentest challenges than either model alone, and the
boost grows as the two models' per-challenge success becomes less correlated.

## What this repo does

- **Models.** Alloy of `claude-opus-4-8` (Anthropic) + `glm-5.3` (Zhipu/Z.ai) —
  a cross-provider pair, the configuration the paper says works best. (The paper
  used Sonnet/Gemini/GPT-4.1; we use what the API keys allow.)
- **Benchmark.** [pwn.college `ctf-archive`](https://github.com/pwncollege/ctf-archive):
  703 real CTF challenges. We auto-triage a clean subset of **PWN** challenges —
  the paper's own domain is exploitation, so this is the closest analog.
- **Sandboxing.** Each solve attempt runs in its own ephemeral **Daytona** sandbox
  built from a prebuilt CTF toolchain snapshot (pwntools, gcc/gdb, etc.).
- **Scoring (no ground-truth needed).** We plant a random `pwn.college{…}` flag at
  `/flag` (root-only) and install the challenge binary **setuid-root**. The agent
  runs as an unprivileged user and *cannot* read `/flag` directly — it must exploit
  the binary to exfiltrate it. Success = the agent submits the exact planted flag.

## Architecture

| File | Role |
|---|---|
| `alloy/transcript.py` | Provider-neutral shared transcript (messages + tool calls/results). |
| `alloy/providers.py`  | Anthropic + OpenAI-compatible (GLM) adapters. Each serializes the *whole* shared history into its own wire format — the alloy crux. |
| `alloy/agent.py`      | ReAct tool-calling loop; `SingleModel` and `Alloy` (random per-step swap) policies; `bash` + `submit_flag` tools. |
| `alloy/sandbox.py`    | Daytona wrapper: root setup vs. non-root `hacker` agent exec. |
| `benchmark/rehost.py` | Install a challenge, plant `/flag`, setuid the binary, verify the flag isn't directly readable. |
| `benchmark/triage.py` | Score + smoke-test candidates → `benchmark/subset.json`. |
| `run.py`              | Parallel orchestrator over (challenge × config × attempt). |
| `analyze.py`          | Success-rate table, either-solves baseline, alloy lift, Spearman correlation. |
| `build_snapshot.py`   | One-time Daytona toolchain snapshot build. |

## Running

```bash
# 0. credentials in .env: ANTHROPIC_API_KEY, GLM_API_KEY, DAYTONA_API_KEY
python build_snapshot.py                 # once: build toolchain snapshot
python benchmark/triage.py --target 24   # select the subset
python run.py --attempts 3 --parallel 30 # the replication run
python analyze.py                        # results -> results/summary.md
```

## Honest limitations

- **Domain narrowed to PWN.** The setuid/planted-flag model gives unambiguous
  scoring without per-challenge ground truth, but only applies cleanly to
  binary-exploitation challenges. Crypto/rev/web are excluded (no automatable
  ground truth in the archive).
- **Not the paper's exact benchmark or models.** XBOW's benchmark and agent are
  proprietary; numbers are not directly comparable. We replicate the *method and
  the effect shape* (alloy > best single; lift tracks model diversity), not the
  exact percentages.
- **Some challenges may be unsolvable as rehosted** (binary never reaches `/flag`).
  This lowers all configs equally, so the alloy-vs-single comparison stays valid.
