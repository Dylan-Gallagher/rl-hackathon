from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from contracts import Transcript, TranscriptMessage, episode_id_for, write_transcript
from path4.scoreboard.metrics import aggregate
from path4.scoreboard.server import create_app

from path4.scoreboard.tests.test_metrics import make_fixture


def test_health_and_root(tmp_path):
    app = create_app(tmp_path, refresh_s=0)
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"] is True
    r = c.get("/")
    assert r.status_code == 200
    assert "scoreboard" in r.text.lower()


def test_summary_matches_metrics(tmp_path):
    for t in make_fixture():
        write_transcript(t, tmp_path / f"{t.policy.replace(':', '_')}.jsonl")
    # race summary in a subdirectory (runs/...) must be picked up
    (tmp_path / "runs" / "race-1").mkdir(parents=True)
    (tmp_path / "runs" / "race-1" / "summary.json").write_text(
        json.dumps({"race_id": "r1", "task_id": "t1", "winner": "solo:test"})
    )
    c = TestClient(create_app(tmp_path, refresh_s=0))
    got = c.get("/api/summary").json()
    want = aggregate(make_fixture(), ks=(1, 4, 8))
    # races differ (only one here); compare policy block built with same ks
    want_policies = {
        p["policy"]: {**p, "race_wins": got_race_wins(got, p["policy"])}
        for p in want["policies"]
    }
    for p in got["policies"]:
        assert p == want_policies[p["policy"]], p["policy"]
    assert got["episodes"] == 22
    assert got["races"] == 1


def got_race_wins(got, policy):
    for p in got["policies"]:
        if p["policy"] == policy:
            return p["race_wins"]
    return None


def test_episode_listing_pagination_and_filters(tmp_path):
    for t in make_fixture():
        write_transcript(t, tmp_path / "a.jsonl")
    c = TestClient(create_app(tmp_path, refresh_s=0))
    all_eps = c.get("/api/episodes?limit=100").json()
    assert all_eps["total"] == 22
    assert len(all_eps["episodes"]) == 22

    page = c.get("/api/episodes?limit=5&offset=5").json()
    assert len(page["episodes"]) == 5
    assert page["total"] == 22

    solo = c.get("/api/episodes?policy=solo:test").json()
    assert solo["total"] == 10
    solved = c.get("/api/episodes?policy=solo:test&solved=true").json()
    assert solved["total"] == 5
    assert all(e["solved"] for e in solved["episodes"])
    # metadata fields present
    e = solved["episodes"][0]
    assert {"episode_id", "task_id", "policy", "solved", "steps", "n_messages", "category"} <= set(e)


def test_episode_detail_has_messages_with_model_labels(tmp_path):
    t = Transcript(
        task_id="t1",
        episode_id=episode_id_for("solo:m", "t1", 0),
        policy="solo:m",
        solved=True,
        steps=3,
        category="pwn",
        messages=[
            TranscriptMessage(turn=0, role="assistant", content="plan", model="anthropic/claude-opus-4"),
            TranscriptMessage(turn=0, role="tool", content="output"),
        ],
    )
    write_transcript(t, tmp_path / "a.jsonl")
    c = TestClient(create_app(tmp_path, refresh_s=0))
    d = c.get(f"/api/episode/{t.episode_id}").json()
    assert d["solved"] is True
    assert d["messages"][0]["model"] == "anthropic/claude-opus-4"
    assert d["messages"][1]["model"] is None
    assert c.get("/api/episode/nope").status_code == 404


def test_new_file_picked_up_between_calls(tmp_path):
    write_transcript(
        Transcript(task_id="t", episode_id="e0", policy="solo:m", solved=True),
        tmp_path / "a.jsonl",
    )
    c = TestClient(create_app(tmp_path, refresh_s=0))
    assert c.get("/api/summary").json()["episodes"] == 1

    time.sleep(0.02)
    write_transcript(
        Transcript(task_id="t", episode_id="e1", policy="solo:m", solved=False),
        tmp_path / "sub" / "b.jsonl",  # nested dir
    )
    assert c.get("/api/summary").json()["episodes"] == 2
    assert c.get("/api/episodes?limit=10").json()["total"] == 2
