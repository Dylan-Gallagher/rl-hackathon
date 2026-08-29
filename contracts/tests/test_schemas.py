"""Schema round-trip tests: §3.1/§3.2 field names must be EXACT."""

import json

from contracts.task import Task, load_tasks
from contracts.transcript import Transcript, TranscriptMessage


def test_task_field_names_exact():
    t = Task(task_id="x", source="custom", category="misc")
    assert list(Task.model_fields) == [
        "task_id", "source", "category", "env", "flag", "prompt", "horizon", "split"]
    assert list(type(t.env).model_fields) == ["image", "launch"]
    assert list(type(t.flag).model_fields) == ["mode", "verify", "format"]
    assert list(type(t.horizon).model_fields) == ["max_steps", "timeout_s"]


def test_task_json_round_trip(tmp_path):
    t = Task(task_id="t1", source="nyuctf", category="rev",
             env={"image": "ghcr.io/x/y:z", "launch": "supervisor"},
             flag={"mode": "static", "verify": "exact", "format": "flag{abc}"},
             prompt="do it", horizon={"max_steps": 10, "timeout_s": 60},
             split="eval")
    p = tmp_path / "t1.json"
    p.write_text(t.model_dump_json())
    t2 = Task.load(p)
    assert t2 == t
    assert Task.from_dict(json.loads(t.model_dump_json())) == t


def test_transcript_field_names_exact():
    tr = Transcript(task_id="t", episode_id="e", policy="solo:m")
    assert list(Transcript.model_fields) == [
        "task_id", "episode_id", "policy", "split", "messages", "solved",
        "steps", "flags_found", "sandbox_id", "tokens_in", "tokens_out"]
    assert list(TranscriptMessage.model_fields) == ["turn", "role", "content", "model"]


def test_transcript_extra_fields_allowed():
    tr = Transcript(task_id="t", episode_id="e", policy="alloy:a,b",
                    category="crypto", race_id="r1")
    assert tr.category == "crypto" and tr.race_id == "r1"


def test_load_tasks_examples():
    tasks = load_tasks("contracts/tasks/examples")
    assert len(tasks) == 3
    assert {t.launch if False else t.env.launch for t in tasks} <= {
        "supervisor", "compose", "repl", "none"}
    assert {t.task_id for t in tasks} == {
        "random-crypto-md5-0001", "nyuctf-2021f-rev-maze", "custom-web-notesql-0001"}
