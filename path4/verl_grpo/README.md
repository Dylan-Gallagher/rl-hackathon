# path4/verl_grpo — GRPO on the learnable band (veRL training side)

Pipeline position: **alloy → traces → SFT (`path4/coldstart`) → EI (Path 3) → GRPO (here)**.
This package implements README Path 4 step 2: binary flag reward, DAPO-style
dynamic sampling, EI pass-rate-band curriculum, and a veRL-shaped async tool
executor over the §3.3 `CTFEnv` protocol.

## What's here

| File | What |
|---|---|
| `reward.py` | `flag_reward` (binary 0/1, flag-as-reward §1.2), `reward_group` → `GroupStats`, `dapo_filter` (drop all-zero/all-one groups) |
| `curriculum.py` | EI stats → learnable band (default 5–40% pass rate, sorted closest-to-20% first) → JSON + rich table |
| `ctf_tool_executor.py` | `CTFToolExecutor`: veRL-shaped async `execute(action, state)`; `run_episode(policy_fn, task)` full generate→step→reward loop |
| `configs/grpo_verl.yaml` | veRL agentic GRPO skeleton (values needing veRL-version confirmation are marked) |
| `run_grpo.sh` | launcher skeleton + fallback ladder (Path 1 Unsloth GRPO if veRL slips) |
| `tests/` | pytest suite (scripted policy solves MockCTFEnv end-to-end) |

## Quickstart

```bash
# curriculum from Path 3 EI stats
python -m path4.verl_grpo.curriculum path3/ei/stats.json path4/verl_grpo/data/curriculum.json

# tests (no torch/verl needed)
python -m pytest path4/verl_grpo -q

# real GRPO (needs veRL installed)
bash path4/verl_grpo/run_grpo.sh
```

## Mocked vs real

- **Real / dependency-free**: reward math, DAPO filter, curriculum, executor logic, full episode loop.
- **Env backends**: `MockCTFEnv` by default; `DaytonaCTFEnv` automatically when `DAYTONA_API_KEY` is set (via `default_env_factory`); any factory can be injected.
- **Stubbed**: the veRL launcher (`run_grpo.sh` `python -m verl...` is a TODO placeholder) and `make_verl_tool_executor` (raises a clean ImportError without veRL). Fallback: Path 1 Unsloth single-turn GRPO is proven.

## Env vars

- `DAYTONA_API_KEY` (optional) — switches the executor's default env factory to `DaytonaCTFEnv`.
