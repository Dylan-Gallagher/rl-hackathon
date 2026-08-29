"""Project full-sweep cost from completed trajectories.

Key idea: the context sent on turn i is the same no matter which model is
called, and each turn's logged input_tokens IS that context size. So summing
every turn's input/output tokens over a trajectory gives the token profile an
identical-length trajectory would have under ANY condition; we then price it
three ways (all-Claude, all-GLM, measured-alloy).
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import config

A = config.MODEL_A["label"]
B = config.MODEL_B["label"]


def price(label, tin, tout):
    p = config.PRICING[label]
    return tin / 1e6 * p["input"] + tout / 1e6 * p["output"]


def profile(traj):
    tin = sum(t["usage"]["input_tokens"] for t in traj["turns"])
    tout = sum(t["usage"]["output_tokens"] for t in traj["turns"])
    return tin, tout, len(traj["turns"])


def main():
    paths = sys.argv[1:] or glob.glob(str(config.TRAJ_DIR / "**" / "*.json"), recursive=True)
    trajs = [json.load(open(p)) for p in paths if Path(p).name != "summary.json"]
    trajs = [t for t in trajs if t.get("turns")]
    if not trajs:
        print("No trajectories found.")
        return

    tin = sum(profile(t)[0] for t in trajs)
    tout = sum(profile(t)[1] for t in trajs)
    turns = sum(profile(t)[2] for t in trajs)
    n = len(trajs)
    avg_turns = turns / n
    avg_tin, avg_tout = tin / n, tout / n

    print(f"Sampled {n} trajectory(ies). Avg turns={avg_turns:.1f}, "
          f"avg input toks/traj={avg_tin:,.0f}, avg output toks/traj={avg_tout:,.0f}")

    # per-trajectory cost under each condition (identical-length assumption)
    a_solo = price(A, avg_tin, avg_tout)
    b_solo = price(B, avg_tin, avg_tout)
    # alloy: split ~50/50 by turns -> half the tokens priced at each model
    alloy = 0.5 * price(A, avg_tin, avg_tout) + 0.5 * price(B, avg_tin, avg_tout)

    print("\nPer-trajectory projected cost:")
    print(f"  A-solo ({A}):  ${a_solo:.3f}")
    print(f"  B-solo ({B}):  ${b_solo:.3f}")
    print(f"  alloy (~50/50):    ${alloy:.3f}")

    C = config.SAMPLE_SIZE
    K = config.ATTEMPTS_PER_CHALLENGE
    per_cond = C * K
    total = per_cond * (a_solo + b_solo + alloy)
    print(f"\nFull sweep = {C} challenges x {K} attempts x 3 conditions = {C*K*3} trajectories")
    print(f"  A-solo   ({per_cond} traj): ${per_cond*a_solo:8.2f}")
    print(f"  B-solo   ({per_cond} traj): ${per_cond*b_solo:8.2f}")
    print(f"  alloy    ({per_cond} traj): ${per_cond*alloy:8.2f}")
    print(f"  --------------------------------------")
    print(f"  TOTAL LLM cost (avg-length):  ${total:8.2f}")

    # worst case: every trajectory runs to the turn cap. Input grows ~linearly
    # per turn, so total tokens scale ~ (cap/avg_turns) for the extra turns; use
    # a simple linear scale on tokens as an upper bound proxy.
    scale = config.TURN_CAP / max(avg_turns, 1)
    total_worst = total * scale
    print(f"  TOTAL LLM cost (if every run hits {config.TURN_CAP}-turn cap, ~{scale:.1f}x): ${total_worst:8.2f}")
    print("\n(Daytona compute is separate but small: ~$0.10/vCPU-hr x 2 vCPU x few min/traj.)")


if __name__ == "__main__":
    main()
