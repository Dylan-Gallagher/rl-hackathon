"""Stratified sample of the CTF-Dojo pool and preparation of local challenge
files for upload into Daytona.

Sampling frame: the 252 CTF-Dojo challenges that have BOTH a stored
`.flag.sha256` (the sha256 of the canonical pwn.college{...} flag) AND a
`flagCheck` oracle binary. These are self-contained and locally solvable, so
every attempt runs in a single sandbox with deterministic verification.

The sample is stratified to match the FULL-POOL category proportions the user
specified (crypto ~35%, ...), even though we draw from the verifiable subset.
"""
from __future__ import annotations
import json, math, random, re, subprocess, hashlib
from collections import defaultdict
from pathlib import Path
import config

FRAME_PATH = Path("/tmp/frame2.json")
MANIFEST = config.CHALLENGES_DIR / "manifest.json"


def largest_remainder(props: dict[str, float], n: int) -> dict[str, int]:
    raw = {c: props[c] * n for c in props}
    base = {c: int(math.floor(raw[c])) for c in props}
    rem = n - sum(base.values())
    order = sorted(props, key=lambda c: raw[c] - base[c], reverse=True)
    for c in order[:rem]:
        base[c] += 1
    return base


def read_first_matching(fs: list[str], event: str, chal: str, pat: str) -> str | None:
    for f in fs:
        if re.search(pat, f, re.IGNORECASE):
            return f
    return None


def main():
    frame = json.load(open(FRAME_PATH))
    by_cat: dict[str, list[str]] = defaultdict(list)
    for k, v in frame.items():
        by_cat[v["category"]].append(k)
    for c in by_cat:
        by_cat[c].sort()  # deterministic base order

    counts = largest_remainder(config.POOL_PROPORTIONS, config.SAMPLE_SIZE)
    print("Target per-category counts:", counts, "total", sum(counts.values()))

    rng = random.Random(config.MASTER_SEED)
    selected: list[str] = []
    for cat, n in counts.items():
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        selected.extend(pool[:n])
    print(f"Selected {len(selected)} challenges.")

    # sparse-checkout selected challenge dirs
    dirs = [f"{frame[k]['event']}/{frame[k]['challenge']}" for k in selected]
    subprocess.run(["git", "-C", str(config.ARCHIVE_DIR), "sparse-checkout", "set", *dirs], check=True)

    manifest = {}
    for k in selected:
        v = frame[k]
        event, chal = v["event"], v["challenge"]
        cdir = config.ARCHIVE_DIR / event / chal
        fs = v["files"]
        sha_file = read_first_matching(fs, event, chal, r"\.?flag\.sha256(\.txt)?$")
        check_file = read_first_matching(fs, event, chal, r"flagcheck")
        sha_val = (cdir / sha_file).read_text().strip().split()[0]
        desc = ""
        dpath = cdir / "DESCRIPTION.md"
        if dpath.exists():
            desc = dpath.read_text(errors="replace").strip()
        # files to place in the sandbox: everything except pure-metadata + the sha file
        skip = {"REHOST.md", sha_file}
        upload_files = [f for f in fs if f not in skip]
        manifest[k] = {
            "id": k,
            "category": v["category"],
            "event": event,
            "challenge": chal,
            "dir": str(cdir),
            "sha256_flag": sha_val,
            "flag_check_file": check_file,
            "description": desc,
            "upload_files": upload_files,
        }
        print(f"  {k:45s} [{v['category']:9s}] files={len(upload_files)} check={check_file}")

    config.CHALLENGES_DIR.mkdir(exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nWrote manifest with {len(manifest)} challenges -> {MANIFEST}")


if __name__ == "__main__":
    main()
