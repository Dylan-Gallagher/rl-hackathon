"""Baseline analysis: per-challenge results + difficulty-band verdict.

Reports (per benchmark): overall solve rate, solve rate by category tag, and the
mechanical breakdown (solved / task_failure / format_failure / infra_error) so
format failures are visible as distinct from task failures. Flags whether the
benchmark sits in the measurable band: clearly ABOVE floor (>~5%) and well BELOW
ceiling (<~90%). Emits per-challenge rows — not just aggregates.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.results, "*.jsonl")))
    if not files:
        print("no baseline results yet."); return
    out = ["# Small-Model CTF Baseline — Results\n"]
    for f in files:
        bench = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f)]
        n = len(rows)
        solved = sum(r["solved"] for r in rows)
        modes = Counter(r.get("outcome") or "?" for r in rows)
        rate = 100 * solved / n if n else 0
        band = ("ABOVE floor, BELOW ceiling — measurable" if 5 <= rate <= 90
                else "AT/NEAR FLOOR — pick an easier band" if rate < 5
                else "NEAR CEILING — pick a harder band")
        out += [f"## {bench}  (n={n})\n",
                f"- **Solve rate: {rate:.1f}%** ({solved}/{n}) → _{band}_",
                f"- Outcome mix: {dict(modes)}",
                f"- Format failures: {modes.get('format_failure',0)}  |  "
                f"task failures: {modes.get('task_failure',0)}  |  "
                f"infra errors: {modes.get('infra_error',0)}"]
        # by tag
        bytag = defaultdict(lambda: [0, 0])
        for r in rows:
            for tg in (r.get("tags") or ["(untagged)"]):
                bytag[tg][0] += r["solved"]; bytag[tg][1] += 1
        out.append("\n| Category | Solve rate |")
        out.append("|---|---|")
        for tg, (s, t) in sorted(bytag.items()):
            out.append(f"| {tg} | {100*s/t:.0f}% ({s}/{t}) |")
        # per-challenge
        out.append("\n| Challenge | tags | solved | outcome | iters | fmt-fails |")
        out.append("|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["id"]):
            fmt = r.get("format", {})
            ff = (fmt.get("no_tool_call_turns", 0) + fmt.get("malformed_tool_calls", 0))
            rl = "loop" if fmt.get("repetition_loop") else ""
            out.append(f"| {r['id']} | {','.join(r.get('tags',[]))} | "
                       f"{'Y' if r['solved'] else '.'} | {r.get('outcome')} | "
                       f"{r['iters']} | {ff}{('/'+rl) if rl else ''} |")
        out.append("")
    txt = "\n".join(out) + "\n"
    p = os.path.join(args.results, "summary.md")
    open(p, "w").write(txt)
    print(txt); print(f"[written to {p}]")


if __name__ == "__main__":
    main()
