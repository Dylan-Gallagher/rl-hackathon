"""Separate MECHANICAL-failure accounting for small models.

Small models often fail mechanically (bad tool syntax, prose instead of a tool
call, repetition loops) rather than from lack of capability. We classify each
episode transcript so a task 'miss' can be attributed to a FORMAT failure vs a
genuine TASK failure — the distinction the baseline is meant to surface.

Works on the transcript dicts saved by the orchestrator (system, messages,
model_sequence, solved, submitted). Pure/offline: no model calls.
"""
from __future__ import annotations
import re

# text that looks like the model *tried* to act but in prose, not via the API
_PROSE_TOOLCALL = re.compile(
    r"(```(?:bash|sh|python)|\bsubmit_flag\s*\(|\"command\"\s*:|<tool_call>|"
    r"<function|\bfunctions\.|\bTool:\s|\bAction:\s|I(?:'ll| will) run)",
    re.IGNORECASE)


def _assistant_turns(t):
    return [m for m in t["messages"] if m["role"] == "assistant"]


def _text(blocks):
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


def _tool_calls(blocks):
    return [b for b in blocks if b.get("type") == "tool_call"]


def classify(t: dict) -> dict:
    """Return per-episode mechanical-failure telemetry + a dominant outcome."""
    turns = _assistant_turns(t)
    n = len(turns)
    stats = {
        "assistant_turns": n,
        "valid_tool_calls": 0,
        "no_tool_call_turns": 0,        # emitted text, no tool call at all
        "prose_tool_call_turns": 0,     # looked like an action but not via API
        "malformed_tool_calls": 0,      # tool call present but unusable args
        "max_cmd_repetition": 0,        # longest run of identical bash commands
        "repetition_loop": False,
    }
    cmd_seq = []
    for m in turns:
        blocks = m["content"]
        tcs = _tool_calls(blocks)
        txt = _text(blocks)
        if not tcs:
            stats["no_tool_call_turns"] += 1
            if _PROSE_TOOLCALL.search(txt or ""):
                stats["prose_tool_call_turns"] += 1
            continue
        for tc in tcs:
            args = tc.get("arguments") or {}
            name = tc.get("name")
            bad = ("_raw" in args
                   or (name == "bash" and not str(args.get("command", "")).strip())
                   or (name == "submit_flag" and not str(args.get("flag", "")).strip())
                   or name not in ("bash", "submit_flag"))
            if bad:
                stats["malformed_tool_calls"] += 1
            else:
                stats["valid_tool_calls"] += 1
            if name == "bash" and not bad:
                cmd_seq.append(str(args.get("command", "")).strip())
    # longest consecutive identical-command run
    run = best = 0
    prev = None
    for c in cmd_seq:
        run = run + 1 if c == prev else 1
        best = max(best, run)
        prev = c
    stats["max_cmd_repetition"] = best
    stats["repetition_loop"] = best >= 4

    solved = bool(t.get("solved"))
    fmt_turns = (stats["no_tool_call_turns"] + stats["malformed_tool_calls"])
    # dominant outcome
    if solved:
        outcome = "solved"
    elif stats["valid_tool_calls"] == 0 and n > 0:
        outcome = "format_failure"          # never issued a single valid tool call
    elif stats["repetition_loop"]:
        outcome = "format_failure"          # stuck repeating
    elif n > 0 and fmt_turns >= 0.5 * n:
        outcome = "format_failure"          # majority of turns mechanically wasted
    else:
        outcome = "task_failure"            # acted correctly, just didn't solve it
    stats["outcome"] = outcome
    return stats


if __name__ == "__main__":  # validate on whatever transcripts exist
    import glob, json, os, sys
    d = sys.argv[1] if len(sys.argv) > 1 else "results/transcripts"
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    print(f"classifying {len(files)} transcripts in {d}\n")
    hdr = f"{'outcome':15} {'turns':>5} {'valid':>5} {'notc':>5} {'prose':>5} {'malf':>4} {'rep':>4}  file"
    print(hdr); print("-" * len(hdr))
    from collections import Counter
    agg = Counter()
    for f in files:
        t = json.load(open(f))
        s = classify(t)
        agg[s["outcome"]] += 1
        print(f"{s['outcome']:15} {s['assistant_turns']:>5} {s['valid_tool_calls']:>5} "
              f"{s['no_tool_call_turns']:>5} {s['prose_tool_call_turns']:>5} "
              f"{s['malformed_tool_calls']:>4} {s['max_cmd_repetition']:>4}  "
              f"{os.path.basename(f)}")
    print("\naggregate:", dict(agg))
