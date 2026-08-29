# Small-Model CTF Baseline (Qwen3-8B, self-served via vLLM)

**Separate** from the alloy replication (own runner, own `baseline/results/`), but
reuses the identical agent harness: `alloy.agent` (same ReAct loop, `bash` +
`submit_flag` tools, same turn cap) and `alloy.providers.OpenAICompatAdapter`
pointed at a **self-hosted vLLM** endpoint. Self-hosting is deliberate: later
fine-tuned variants exist on no API, and holding the serving stack fixed (chat
template, sampling, tokenizer, tool-call parser) keeps them from confounding
comparisons against this baseline.

## Status: built & validated except the GPU run (blocked on Daytona GPU credits)

Daytona GPU sandboxes are supported on this account but creation fails with
`Organization doesn't have GPU credits. Add more by visiting the Wallet page`.
That is a billing action only you can take. Everything else is built and the
non-GPU parts are validated:

| Piece | State |
|---|---|
| `serve_vllm.py` — ephemeral GPU sandbox + vLLM serving Qwen3-8B, pinned template/sampling/tool-parser | built; **untested** (needs GPU credits) |
| `intercode.py` — InterCode-CTF loader + rehost (100 tasks fetched; solution files excluded) | **validated** (data fetched, no answer leakage) |
| `run_baseline.py` — baseline orchestrator, reuses harness, per-challenge + format telemetry | built; syntax/import-checked |
| `format_failures.py` — mechanical-failure classifier | **validated** on 17 real frontier transcripts (they show 0 format failures, as expected) |
| `analyze_baseline.py` — per-challenge + per-tag + band verdict | built |

## Run (once GPU credits are added)

```bash
python baseline/serve_vllm.py                         # -> baseline/results/serving.json
python baseline/run_baseline.py --benchmark intercode     # the signal band
python baseline/run_baseline.py --benchmark archive_pwn   # the hard anchor
python baseline/analyze_baseline.py                   # -> baseline/results/summary.md
```
Fine-tuned variants: `BASELINE_MODEL=/path/or/hf-id` and (if needed)
`BASELINE_CHAT_TEMPLATE=/path/to/template.jinja` — nothing else changes.

## Where to evaluate (the point of this run)

Published Qwen3-8B: ~46% InterCode-CTF, ~5% Cybench, <1% NYU CTF. So the goal is
to evaluate on the band that is clearly **above floor** and **below ceiling**,
where fine-tuning has room to move the number.

- **InterCode-CTF (integrated, available)** — picoCTF-derived; the ~46% signal
  band. Stratify by the built-in tags to find sub-bands (expect General Skills /
  Crypto / RE higher; Forensics / Binary Exploitation lower). This is the primary
  measurement surface.
- **pwn.college ctf-archive PWN (integrated, available)** — real competition pwn,
  ≈ Cybench/NYU hardness. Use as the near-floor anchor (expect <5%); good for
  confirming the ceiling of difficulty, not for measuring incremental gains.
- **CTF-DOJO (pluggable, NOT integrated)** — HF dataset is gated (`401`); add an
  `HF_TOKEN` and a loader mirroring `intercode.py` to get mid-band, difficulty-
  labelled granularity between the two anchors above.

Recommendation: **land the signal in InterCode-CTF (overall + by-tag), anchor the
floor with archive_pwn, and add CTF-DOJO easy/medium tiers for mid-band
resolution once HF access is granted.**

## Format vs task failure

`format_failures.classify()` labels each episode `solved` / `task_failure` /
`format_failure` (never issued a valid tool call, majority of turns mechanically
wasted, or a repetition loop), plus raw counts (no-tool-call turns, prose-instead-
of-call, malformed args, longest identical-command run). Infra/serving errors are
tracked separately in each result's `error` field (`infra_error` outcome), so an
8B's mechanical failures are never mistaken for genuine capability failures.
