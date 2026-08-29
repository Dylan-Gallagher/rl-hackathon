# Path 4 — Full Agentic GRPO + Alloy Ensemble (flagship)

Implementation of **PATH 4** from the [root README](../README.md) §4: one open model RL'd
with multi-turn GRPO inside the CTF gym, bootstrapped by alloy traces, deployed inside a
racing alloy ensemble — scored live.

```
                ┌── per-turn alloy (frontier APIs) ──> success traces ─┐
                │                                                            ▼
 student ── cold-start SFT ──> EI rounds (Path 3) ──> GRPO (veRL) ──> student'
                                                                             │
 live demo: student' + alloy members racing per challenge ───────────────────┘
 scoreboard: per-model & ensemble Pass@k, per-category, trajectory viewer
```

## Layout

| Component | Dir | What it does | Status |
|---|---|---|---|
| Shared contracts | [`contracts/`](../contracts/) | §3 task schema, transcript JSONL, `CTFEnv` protocol + Mock/Repl/Docker/Daytona backends, flag gen/verify, LiteLLM `models.yaml` | **real** (Daytona SDK internals = tracked TODOs) |
| Cold-start SFT | [`coldstart/`](coldstart/) | `build_dataset` (solved-only, assistant-only mask, caps, dedup) + TRL LoRA `train_sft` (`--dry-run` offline) | core real; TRL wiring guarded (`pip install -e '.[train]'`) |
| GRPO pieces | [`verl_grpo/`](verl_grpo/) | binary flag reward, DAPO dynamic-sampling filter, 5–40% curriculum band, `CTFToolExecutor` (veRL-shaped async adapter over any `CTFEnv`), `grpo_verl.yaml` + launcher | executor/curriculum real; veRL launcher = skeleton w/ fallback ladder |
| Racing ensemble | [`ensemble/`](ensemble/) | agent loop, per-turn alloy routing (`solo:M` / `alloy:M1:w,M2:w`), race k policies on isolated envs — first verified flag wins; shared-findings bus (flag-scrubbed) | **real** (any OpenAI-compatible endpoint) |
| Live scoreboard | [`scoreboard/`](scoreboard/) | Pass@k (unbiased), Maj@k, per-category, race wins; FastAPI server + single-file trajectory-viewer UI; demo corpus seeder | **real** |

## Quickstart (offline, no API keys)

```bash
# 1) full race on the example tasks with scripted mock models
python -m path4.ensemble.cli race \
  --tasks contracts/tasks/examples --policies solo:mock-a,solo:mock-b \
  --mock --env mock --out runs/demo_race --max-steps 10

# 2) scoreboard over that output (also serves the race results)
python -m path4.scoreboard.cli table --transcripts runs/demo_race

# 3) richer demo corpus + live UI on http://localhost:8080
#    (check the port is free first — e.g. on this host 8080 is taken, use --port 8099)
python -m path4.scoreboard.demo.seed_demo --out runs/demo
python -m path4.scoreboard.cli serve --transcripts runs/demo --port 8080
```

## Real runs

1. Stand up the LiteLLM proxy with [`contracts/models.yaml`](../contracts/models.yaml)
   (alloy group + solo deployments; needs `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`).
2. Point the ensemble at it:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1 OPENAI_API_KEY=sk-...
python -m path4.ensemble.cli race \
  --tasks <task-dir> \
  --policies solo:anthropic/claude-sonnet-4,alloy:anthropic/claude-sonnet-4:0.6,google/gemini-2.5-pro:0.4,solo:student-qwen \
  --env mock|repl|docker|daytona --out runs/live --max-steps 40
```

The `solo:student-*` policy is the RL'd student racing frontier models — the money shot.

## Training pipeline

```bash
# cold-start: alloy success traces -> masked SFT dataset -> LoRA SFT
python -m path4.coldstart.build_dataset runs/live -o data/sft.jsonl --val-frac 0.05
python -m path4.coldstart.train_sft --dataset data/sft.jsonl --config path4/coldstart/configs/sft_lora.yaml --dry-run

# after Path 3 EI rounds emit stats: pick the learnable band (5–40% pass rate)
python -m path4.verl_grpo.curriculum data/ei_stats.json data/curriculum.json

# GRPO: see path4/verl_grpo/README.md + run_grpo.sh (veRL skeleton)
```

## Fallback ladder (graceful degradation)

per root README §4 risks:

1. **veRL multi-turn GRPO** slips →
2. **Unsloth single-turn GRPO** (Path 1 result already proves GRPO-on-CTF) →
3. **EI-only** self-improvement (Path 3 harness, this scoreboard still shows it) →
4. **Alloy-only** results (Path 2 baselines; ensemble + scoreboard run unchanged).

The scoreboard and ensemble are independent of the training stack — the live demo
degrades to frontier-model racing if the student isn't ready.

## Anti-cheat (§1.5)

- Fresh `flag{uuid4}` per episode (task-identity-mixed seeded variants for reproducible runs).
- Flag verification runs **outside** sandboxes (`contracts.flag`); envs only expose outputs.
- Fresh sandbox per episode (`env_factory(task)` per racer episode); `env.close()` guaranteed.
- Egress lock on Daytona backends (placeholder call — wire before real training runs).
- Findings bus scrubs any `flag{...}` before sharing between teammates.
- No cross-episode state: mock client resets per episode; re-runs overwrite, never double-count.

## Tests

```bash
python -m pytest contracts path4 -q   # 91 tests, fully offline
```
