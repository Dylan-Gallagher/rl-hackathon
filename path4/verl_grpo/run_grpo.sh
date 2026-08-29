#!/usr/bin/env bash
# Path 4 GRPO launch script (veRL agentic multi-turn GRPO).
# SKELETON: the `python -m verl...` invocation is a placeholder — see TODOs.
#
# Environment: use uv (fast, reproducible):
#   uv venv && source .venv/bin/activate
#   uv pip install -e '.[train]' verl vllm   # veRL extras; torch comes with verl
# (or: python -m venv .venv && .venv/bin/pip install ...)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

CONFIG=path4/verl_grpo/configs/grpo_verl.yaml
CURRICULUM=path4/verl_grpo/data/curriculum.json

# 1) Build the curriculum from Path 3's EI stats (pure python, no GPU deps).
python -m path4.verl_grpo.curriculum path3/ei/stats.json "$CURRICULUM"

# 2) Launch veRL agentic GRPO.
# TODO(train): replace with the installed veRL version's actual entrypoint,
#   e.g. `python -m verl.trainer.main_ppo --config-path ... --config-name ...`
#   or the agentic recipe launcher (verl agentic docs), passing:
#     - actor model path  = runs/sft_coldstart (Path 4 step 1 output)
#     - tool executor     = path4.verl_grpo.ctf_tool_executor:CTFToolExecutor
#     - curriculum file   = $CURRICULUM
#     - group_size 8-16, binary reward, dynamic sampling filter ON
echo "PLACEHOLDER: python -m verl.trainer.main_ppo --config $CONFIG --curriculum $CURRICULUM"

# FALLBACK LADDER (README Path 4): if veRL integration slips →
#   Path 1 Unsloth single-turn GRPO is the proven fallback
#   (HackSynth-GRPO recipe, Llama-3.1-8B 0.1→0.9 Pass@8), then EI-only (Path 3).
