"""Example tasks round-trip through the real loaders (no network)."""

from contracts.task import Task, load_tasks

EXAMPLES = "contracts/tasks/examples"


def test_examples_load():
    tasks = {t.task_id: t for t in load_tasks(EXAMPLES)}
    assert set(tasks) == {
        "random-crypto-md5-0001", "nyuctf-2021f-rev-maze", "custom-web-notesql-0001"}
    assert tasks["random-crypto-md5-0001"].env.launch == "repl"
    assert tasks["random-crypto-md5-0001"].flag.mode == "generated"
    assert tasks["nyuctf-2021f-rev-maze"].flag.mode == "static"
    assert tasks["nyuctf-2021f-rev-maze"].flag.verify == "exact"
    assert tasks["custom-web-notesql-0001"].flag.verify == "regex"
    assert tasks["custom-web-notesql-0001"].env.launch == "compose"
    assert tasks["custom-web-notesql-0001"].split == "eval"


def test_example_files_valid_pydantic():
    import json
    from pathlib import Path
    for fp in sorted(Path(EXAMPLES).glob("*.json")):
        Task.model_validate(json.loads(fp.read_text()))
