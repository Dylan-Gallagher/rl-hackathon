"""Analyze results/runs.jsonl -> the paper's headline table + correlation.

Computes, per config, the overall success rate (fraction of attempts solved)
and per-challenge solve rate. Then:
  * either-solves baseline: a challenge counts solved if EITHER single model
    solved it on the matched attempt (the paper's 'both single' column).
  * alloy lift vs the best single model.
  * Spearman correlation between the two single models' per-challenge solve
    rates (the paper's 'lower correlation -> bigger alloy boost' analysis).
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "runs.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary.md"))
    args = ap.parse_args()
    rows = load(args.runs)
    if not rows:
        print("no results yet"); return

    configs = sorted({r["config"] for r in rows})
    challenges = sorted({r["cid"] for r in rows})

    # attempts[config][cid] = list of bools
    att = defaultdict(lambda: defaultdict(list))
    for r in rows:
        att[r["config"]][r["cid"]].append(bool(r["solved"]))

    def overall_rate(cfg):
        s = [v for cid in att[cfg] for v in att[cfg][cid]]
        return 100.0 * sum(s) / len(s) if s else 0.0

    def per_challenge_rate(cfg):
        return {cid: (sum(v) / len(v) if v else 0.0)
                for cid, v in att[cfg].items()}

    lines = ["# Alloy Agents — Replication Results\n"]
    lines.append(f"Challenges: {len(challenges)} | configs: {configs} | "
                 f"total runs: {len(rows)}\n")

    # headline table
    lines.append("## Success rate by configuration\n")
    lines.append("| Configuration | Success rate |")
    lines.append("|---|---|")
    label = {"single_opus": "claude-opus-4-8 (single)",
             "single_sonnet": "claude-sonnet-4-5 (single)",
             "alloy": "Alloy (opus-4-8 + sonnet-4-5)"}
    for cfg in ["single_sonnet", "single_opus", "alloy"]:
        if cfg in att:
            lines.append(f"| {label.get(cfg, cfg)} | {overall_rate(cfg):.1f}% |")

    # either-solves baseline (per challenge: solved if either single ever solved)
    if "single_opus" in att and "single_sonnet" in att:
        o = per_challenge_rate("single_opus")
        g = per_challenge_rate("single_sonnet")
        common = sorted(set(o) & set(g))
        either = [1.0 if (o[c] > 0 or g[c] > 0) else 0.0 for c in common]
        either_rate = 100.0 * sum(either) / len(either) if either else 0.0
        # insert the baseline row conceptually (report separately)
        lines.append(f"| Either single solves (baseline) | {either_rate:.1f}% |")

        # Spearman between the two single models' per-challenge solve rates
        ov = [o[c] for c in common]
        gv = [g[c] for c in common]
        rho, p = spearmanr(ov, gv)
        lines.append("")
        lines.append("## Model diversity\n")
        lines.append(f"- Spearman correlation (opus vs sonnet per-challenge solve "
                     f"rate): **{rho:.3f}** (p={p:.3f}, n={len(common)})")

        best_single = max(overall_rate("single_opus"), overall_rate("single_glm"))
        if "alloy" in att:
            lift = overall_rate("alloy") - best_single
            lines.append(f"- Best single model: **{best_single:.1f}%**")
            lines.append(f"- Alloy: **{overall_rate('alloy'):.1f}%**")
            lines.append(f"- **Alloy lift over best single: {lift:+.1f} pts**")

    # per-challenge detail
    lines.append("\n## Per-challenge solve rate\n")
    lines.append("| Challenge | " + " | ".join(configs) + " |")
    lines.append("|" + "---|" * (len(configs) + 1))
    rates = {cfg: per_challenge_rate(cfg) for cfg in configs}
    for cid in challenges:
        cells = [f"{100*rates[cfg].get(cid,0):.0f}%" for cfg in configs]
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")

    txt = "\n".join(lines) + "\n"
    open(args.out, "w").write(txt)
    print(txt)
    print(f"[written to {args.out}]")


if __name__ == "__main__":
    main()
