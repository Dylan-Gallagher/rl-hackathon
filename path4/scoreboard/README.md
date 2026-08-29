# Path 4 Scoreboard — live Pass@k / Maj@k over canonical transcripts

It's just transcripts + a viewer. The scoreboard aggregates every
`*.jsonl` transcript under a directory (recursively — including `runs/`
race subdirs) into per-policy / per-category **Pass@k** and **Maj@k**, and
serves a live-updating dark-theme web UI with a per-episode trajectory
viewer. No database, no build step: rescan-on-mtime + one static HTML file.

## Quickstart

```bash
# 1) seed the offline demo corpus (deterministic, 5 policies incl. student-grpo)
python -m path4.scoreboard.demo.seed_demo --out runs/demo

# 2) serve the scoreboard
python -m path4.scoreboard.cli serve --transcripts runs/demo --port 8080
#   (equivalently: python -m path4.scoreboard.server --transcripts runs/demo --port 8080)

# 3) open http://127.0.0.1:8080
```

CLI extras:

```bash
python -m path4.scoreboard.cli table --transcripts runs/demo        # rich console table
python -m path4.scoreboard.cli export --transcripts runs/demo --json summary.json
```

## How the ensemble writes into it

Just append canonical transcripts (README §3.2) anywhere under the scanned
directory, one JSON line per episode — `contracts.write_transcript` does it.
The server rescans whenever the newest file mtime changes (checked at most
once per `--refresh` seconds, 3s default; the UI polls `/api/summary` every
3s), so new episodes appear live.

Races: if the tree contains `summary.json` files in the path4/ensemble
format (`{"race_id", "task_id", "winner", ...}`), each counts as one race
win for `winner` and adds a "race W" column.

## Metrics (see `metrics.py` docstring for the full definitions)

- **Pass@1** = solves / episodes (raw solve rate).
- **Pass@k (k>1)** = unbiased combinatorial estimator `1 - C(n-c,k)/C(n,k)`
  per (policy, task), averaged over tasks with ≥ k episodes; tasks with
  fewer episodes are excluded (guarded — no unbiased estimate exists).
- **Maj@k** = task solved iff > k/2 of its first k episodes (by episode_id)
  are solved; averaged over tasks with ≥ k episodes.
- Per-category breakdown via the optional `category` transcript extension
  (fallback `unknown`); avg steps/tokens solved-vs-unsolved.
- First-solve-time is **not** computed — the transcript schema has no
  wall-clock timing.

## API

| endpoint | purpose |
|---|---|
| `GET /` | static UI (`static/index.html`) |
| `GET /api/health` | liveness + episode count |
| `GET /api/summary` | full aggregate (same shape as `metrics.aggregate`) |
| `GET /api/episodes?policy&task&solved&category&limit&offset` | episode metadata, newest-first |
| `GET /api/episode/{id}` | full messages incl. per-turn `model` label (trajectory viewer payload) |

## Demo tips (projector)

- Polls every 3s — leave it on screen while the ensemble writes episodes;
  rows re-sort by Pass@8 live.
- Click a policy row → slide-over episode list; click an episode → full
  trajectory (role-colored, model label per turn, long tool output
  collapsible, solved badge + flags). Esc closes.
- `python -m path4.scoreboard.demo.seed_demo` with no args writes a small
  (~40-episode) committed corpus to `path4/scoreboard/demo/data` for a
  repo-checkout demo without running anything.

## Tests

`python -m pytest path4/scoreboard -q` — hand-computed Pass@k/Maj@k fixtures
(edge cases: task with <k episodes, all-solve, none-solve, Maj tie), server
rescan/pagination/detail via `fastapi.testclient`.
