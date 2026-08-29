"""Transcript JSONL write/iter + episode_id helper tests."""

from contracts.transcript import (
    Transcript,
    TranscriptMessage,
    episode_id_for,
    iter_transcripts,
    write_transcript,
)


def _make():
    return Transcript(
        task_id="t1", episode_id="e1", policy="alloy:w1,w2", split="train",
        messages=[
            TranscriptMessage(turn=0, role="assistant", content="hello\nmultiline",
                              model="anthropic/claude-sonnet-4"),
            TranscriptMessage(turn=0, role="tool", content="ok", model=None),
        ],
        solved=True, steps=2, flags_found=["flag{abc}"], sandbox_id="s-1",
        tokens_in=100, tokens_out=50)


def test_write_transcript_one_line(tmp_path):
    p = tmp_path / "tr.jsonl"
    write_transcript(_make(), p)
    write_transcript(_make(), p)  # append
    lines = p.read_text().splitlines()
    assert len(lines) == 2


def test_write_iter_round_trip(tmp_path):
    p = tmp_path / "tr.jsonl"
    t = _make()
    write_transcript(t, p)
    loaded = list(iter_transcripts(p))
    assert len(loaded) == 1
    assert loaded[0] == t
    # dir mode picks up *.jsonl
    assert len(list(iter_transcripts(tmp_path))) == 1


def test_iter_skips_malformed(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"task_id": "t", "episode_id": "e", "policy": "solo:m"}\n'
                 "not json\n")
    out = list(iter_transcripts(p))
    assert len(out) == 1 and out[0].task_id == "t"


def test_episode_id_for():
    eid = episode_id_for("alloy:w1,w2", "nyuctf-2021f-rev-maze{uuid4}", 3)
    # format is '{policy}:{task_id}:{idx}', sanitized (':' and '{...}' not filename-safe)
    assert eid == "alloy-w1-w2-nyuctf-2021f-rev-maze-3"
