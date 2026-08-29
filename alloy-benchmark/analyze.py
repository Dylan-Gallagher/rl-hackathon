"""Aggregate the sweep: solve rates per condition + the A/B solo overlap that
tells "the alloy mechanism didn't help" apart from "the two models are too
similar to benefit from mixing".
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import config

COND_LABELS = {"A": config.MODEL_A["label"], "B": config.MODEL_B["label"], "alloy": "alloy (A/B random per turn)"}


def load_all() -> dict:
    out = defaultdict(lambda: defaultdict(list))  # condition -> cid -> [traj,...]
    for cond_dir in config.TRAJ_DIR.iterdir():
        if not cond_dir.is_dir():
            continue
        for f in cond_dir.glob("*.json"):
            t = json.load(open(f))
            out[t["condition"]][t["challenge_id"]].append(t)
    return out


def solved_set(data, cond) -> set:
    """challenges solved in >=1 attempt under a condition."""
    return {cid for cid, trajs in data[cond].items() if any(t["solved"] for t in trajs)}


def attempt_stats(data, cond):
    trajs = [t for cid in data[cond] for t in data[cond][cid]]
    n = len(trajs)
    ns = sum(t["solved"] for t in trajs)
    return n, ns


def main():
    data = load_all()
    manifest = json.load(open(config.CHALLENGES_DIR / "manifest.json"))
    all_cids = set(manifest)
    cats = {cid: manifest[cid]["category"] for cid in manifest}

    print("=" * 72)
    print("AGGREGATE SOLVE RATES")
    print("=" * 72)
    print(f"{'condition':32s} {'solve@k':>9s} {'attempts':>10s} {'per-attempt':>12s} {'cost':>9s}")
    solved = {}
    for cond in ["A", "B", "alloy"]:
        if cond not in data:
            continue
        s = solved_set(data, cond)
        solved[cond] = s
        n_att, n_solved_att = attempt_stats(data, cond)
        cost = sum(t.get("cost_usd", 0) for cid in data[cond] for t in data[cond][cid])
        k = len(all_cids)
        print(f"{COND_LABELS[cond]:32s} {len(s):>4d}/{k:<4d} "
              f"{n_solved_att:>4d}/{n_att:<5d} {100*n_solved_att/max(n_att,1):>10.1f}% ${cost:>7.2f}")

    if "A" in solved and "B" in solved:
        A, B = solved["A"], solved["B"]
        both = A & B
        a_only = A - B
        b_only = B - A
        neither = all_cids - (A | B)
        union = A | B
        jac = len(both) / max(len(union), 1)
        print("\n" + "=" * 72)
        print("PER-CHALLENGE SOLO OVERLAP  (A = %s, B = %s)" % (config.MODEL_A["label"], config.MODEL_B["label"]))
        print("=" * 72)
        print(f"  solved by BOTH:      {len(both):2d}   {sorted(both)}")
        print(f"  solved by A only:    {len(a_only):2d}   {sorted(a_only)}")
        print(f"  solved by B only:    {len(b_only):2d}   {sorted(b_only)}")
        print(f"  solved by NEITHER:   {len(neither):2d}")
        print(f"  union (A or B):      {len(union):2d}")
        print(f"  Jaccard overlap:     {jac:.2f}   "
              f"(low => complementary models; high => redundant models)")

        if "alloy" in solved:
            AL = solved["alloy"]
            print("\n" + "=" * 72)
            print("ALLOY vs SOLO")
            print("=" * 72)
            best_solo = max(len(A), len(B))
            print(f"  alloy solved:            {len(AL)}/{len(all_cids)}")
            print(f"  best single model:       {best_solo}/{len(all_cids)}")
            print(f"  union of both solos:     {len(union)}/{len(all_cids)}  (ceiling if alloy just routes)")
            print(f"  alloy - best_solo:       {len(AL)-best_solo:+d}")
            print(f"  alloy solved that NEITHER solo did: {sorted(AL - union)}")
            print(f"  challenges a solo solved but alloy missed: {sorted(union - AL)}")

    # per-category breakdown
    print("\n" + "=" * 72)
    print("SOLVE@k BY CATEGORY")
    print("=" * 72)
    conds = [c for c in ["A", "B", "alloy"] if c in data]
    header = f"{'category':10s} " + " ".join(f"{COND_LABELS[c][:14]:>15s}" for c in conds)
    print(header)
    for cat in ["crypto", "pwn", "rev", "misc", "forensics", "web"]:
        cat_cids = {cid for cid in all_cids if cats[cid] == cat}
        if not cat_cids:
            continue
        row = f"{cat:10s} "
        for c in conds:
            s = len(solved.get(c, set()) & cat_cids)
            row += f"{s:>7d}/{len(cat_cids):<7d}"
        print(row)

    # alloy per-turn attribution
    if "alloy" in data:
        A_turns = B_turns = 0
        for cid in data["alloy"]:
            for t in data["alloy"][cid]:
                A_turns += t["model_turn_counts"]["A"]
                B_turns += t["model_turn_counts"]["B"]
        tot = A_turns + B_turns
        print("\n" + "=" * 72)
        print("ALLOY PER-TURN ATTRIBUTION")
        print("=" * 72)
        print(f"  A ({config.MODEL_A['label']}): {A_turns} turns ({100*A_turns/max(tot,1):.1f}%)")
        print(f"  B ({config.MODEL_B['label']}): {B_turns} turns ({100*B_turns/max(tot,1):.1f}%)")

    summary = {
        "conditions": {c: sorted(solved.get(c, set())) for c in solved},
    }
    json.dump(summary, open(config.TRAJ_DIR / "summary.json", "w"), indent=2)
    print(f"\nWrote {config.TRAJ_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
