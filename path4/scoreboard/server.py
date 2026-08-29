"""FastAPI live scoreboard over a transcripts directory.

``create_app(transcripts_dir, refresh_s=3)`` — GET / serves the static UI;
``/api/summary`` aggregates the tree, rescanning only when the newest file mtime
changes (cheap cache). ``/api/episodes`` and ``/api/episode/{id}`` back the
trajectory viewer.

Run: ``python -m path4.scoreboard.server --transcripts runs/demo --port 8080``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from path4.scoreboard.metrics import aggregate, scan_race_summaries, scan_transcripts

STATIC_DIR = Path(__file__).parent / "static"


def _episode_meta(t, source_mtime: float) -> dict[str, Any]:
    return {
        "episode_id": t.episode_id,
        "task_id": t.task_id,
        "policy": t.policy,
        "solved": bool(t.solved),
        "steps": t.steps,
        "n_messages": len(t.messages),
        "category": getattr(t, "category", None) or "unknown",
        "flags_found": list(t.flags_found),
        "tokens_in": t.tokens_in,
        "tokens_out": t.tokens_out,
        "sort_time": source_mtime,
    }


def create_app(transcripts_dir: str | Path, refresh_s: float = 3.0) -> FastAPI:
    """App factory. ``transcripts_dir`` is scanned recursively at request time."""
    root = Path(transcripts_dir).resolve()

    state: dict[str, Any] = {
        "mtime": None,
        "summary": None,
        "episodes": [],  # list[dict] episode metadata, newest-first
        "by_id": {},  # episode_id -> Transcript
        "checked": 0.0,
    }

    def _tree_mtime() -> float:
        mtime = 0.0
        if root.is_dir():
            for fp in root.rglob("*"):
                if fp.is_file():
                    try:
                        mtime = max(mtime, fp.stat().st_mtime)
                    except OSError:
                        continue
        return mtime

    def _rescan() -> None:
        transcripts = scan_transcripts(root)
        races = scan_race_summaries(root)
        state["summary"] = aggregate(transcripts, races)
        metas = []
        by_id: dict[str, Any] = {}
        for fp in sorted(root.rglob("*.jsonl")):
            try:
                mtime = fp.stat().st_mtime
            except OSError:
                mtime = 0.0
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    t = _parse(line)
                except Exception:
                    continue
                metas.append(_episode_meta(t, mtime))
                by_id[t.episode_id] = t
        metas.sort(key=lambda m: (m["sort_time"], m["episode_id"]), reverse=True)
        for m in metas:
            m.pop("sort_time", None)
        state["episodes"] = metas
        state["by_id"] = by_id
        state["mtime"] = _tree_mtime()

    def _parse(line: str):
        from contracts import Transcript

        return Transcript.model_validate_json(line)

    def _ensure_fresh() -> None:
        now = time.time()
        if state["summary"] is None or now - state["checked"] >= refresh_s:
            state["checked"] = now
            if _tree_mtime() != state["mtime"] or state["summary"] is None:
                _rescan()

    app = FastAPI(title="path4 scoreboard", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "transcripts_dir": str(root), "episodes": len(state["episodes"])}

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        _ensure_fresh()
        return state["summary"] or {"ks": [], "episodes": 0, "races": 0, "policies": []}

    @app.get("/api/episodes")
    def episodes(
        policy: str | None = None,
        task: str | None = None,
        solved: bool | None = None,
        category: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        _ensure_fresh()
        rows = state["episodes"]
        if policy:
            rows = [r for r in rows if r["policy"] == policy]
        if task:
            rows = [r for r in rows if r["task_id"] == task]
        if solved is not None:
            rows = [r for r in rows if r["solved"] == solved]
        if category:
            rows = [r for r in rows if r["category"] == category]
        return {"total": len(rows), "offset": offset, "limit": limit, "episodes": rows[offset : offset + limit]}

    @app.get("/api/episode/{episode_id}")
    def episode(episode_id: str) -> dict[str, Any]:
        _ensure_fresh()
        t = state["by_id"].get(episode_id)
        if t is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return {
            "episode_id": t.episode_id,
            "task_id": t.task_id,
            "policy": t.policy,
            "split": t.split,
            "solved": bool(t.solved),
            "steps": t.steps,
            "flags_found": list(t.flags_found),
            "category": getattr(t, "category", None) or "unknown",
            "tokens_in": t.tokens_in,
            "tokens_out": t.tokens_out,
            "messages": [m.model_dump() for m in t.messages],
        }

    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="path4 scoreboard server")
    ap.add_argument("--transcripts", required=True, help="transcripts directory (scanned recursively)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--refresh", type=float, default=3.0, help="rescan interval seconds")
    args = ap.parse_args()
    uvicorn.run(create_app(args.transcripts, refresh_s=args.refresh), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
