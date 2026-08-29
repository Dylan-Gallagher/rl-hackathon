"""Offline tests for path4.ensemble (no network; asyncio.run in sync tests)."""

import json
import random
from pathlib import Path

import pytest

from contracts.env.mock import MockCTFEnv
from contracts.task import Task
from contracts.transcript import Transcript

from path4.ensemble.agent import extract_command, extract_findings, run_episode
from path4.ensemble.llm import ChatClient, MockChatClient
from path4.ensemble.policies import AlloyPolicy, SoloPolicy, parse_policy
from path4.ensemble.racer import FindingsBus, make_env_factory, race, scrub_flags


def make_task(task_id: str = "mock-task-1") -> Task:
    return Task(
        task_id=task_id,
        source="custom",
        category="misc",
        prompt="Grab the flag from flag.txt.",
    )


SOLVE_SCRIPT = [
    "```bash\nls\n```",
    "```bash\ncat flag.txt\n```",
    "Got it. FLAG: flag{...}",
]


def test_parse_policy() -> None:
    s = parse_policy("solo:gpt-5.4")
    assert isinstance(s, SoloPolicy)
    assert s.name() == "solo:gpt-5.4"
    a = parse_policy("alloy:claude-sonnet-4:0.6,gemini-2.5-pro:0.4")
    assert isinstance(a, AlloyPolicy)
    assert a.name() == "alloy:claude-sonnet-4:0.6,gemini-2.5-pro:0.4"
    with pytest.raises(ValueError):
        parse_policy("nonsense")
    with pytest.raises(ValueError):
        parse_policy("alloy:only-one")


def test_policy_weighting_chi_square_lite() -> None:
    pol = parse_policy("alloy:a:0.8,b:0.2")
    rng = random.Random(1234)
    n = 2000
    import asyncio

    async def sample() -> list[str]:
        return [await pol.pick_model(t, rng) for t in range(n)]

    counts = {"a": 0, "b": 0}
    for p in asyncio.run(sample()):
        counts[p] += 1
    # 80/20 of 2000 -> expect 1600/400; tolerance +-80 (4 sigma-ish for binomial)
    assert abs(counts["a"] - 1600) <= 80, counts
    assert abs(counts["b"] - 400) <= 80, counts


def test_extract_command_fenced() -> None:
    cmd, nudge = extract_command("Thinking...\n```bash\n$ ls -la\n```\ntrailing talk")
    assert cmd == "ls -la" and nudge == ""


def test_extract_command_sh_fence() -> None:
    cmd, _ = extract_command("```sh\npython3 -c 'print(1)'\n```")
    assert cmd == "python3 -c 'print(1)'"


def test_extract_command_unfenced_single_line() -> None:
    cmd, nudge = extract_command("ls")
    assert cmd == "ls" and nudge == ""


def test_extract_command_garbage_nudges() -> None:
    cmd, nudge = extract_command("I think we should\nlook around first maybe?")
    assert cmd is None and "fenced" in nudge
    cmd, nudge = extract_command("")
    assert cmd is None and nudge


def test_extract_findings() -> None:
    notes = extract_findings("FINDING: binary is stripped, try strings\n```bash\nstrings a.out\n```")
    assert notes == ["binary is stripped, try strings"]
    assert extract_findings("nothing here") == []


def test_mock_chat_client_scripted() -> None:
    mc = MockChatClient(scripts={"m": ["one", "two"]})
    out = asyncio_run_seq(mc, [("hello", "m"), ("hello", "m"), ("hello", "m")])
    assert out == ["one", "two", "two"]  # last repeats
    assert mc.calls[0] == ("m", "one")


def asyncio_run_seq(client: MockChatClient, calls: list[tuple[str, str]]) -> list[str]:
    import asyncio

    async def go() -> list[str]:
        res = []
        for msg, model in calls:
            r = await client.chat([{"role": "user", "content": msg}], model=model)
            res.append(r["content"])
        return res

    return asyncio.run(go())


def test_chat_client_parses_and_served_model() -> None:
    import asyncio

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "served/deployment-x",
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "_requested": body["model"],
        })

    client = ChatClient(base_url="http://test/v1", api_key="k", retries=1,
                        transport=httpx.MockTransport(handler))
    out = asyncio.run(client.chat([{"role": "user", "content": "x"}], model="alloy"))
    assert out == {"content": "hi", "model": "served/deployment-x", "tokens_in": 5, "tokens_out": 2}


def test_episode_solves_and_logs_models() -> None:
    import asyncio

    task = make_task()
    mc = MockChatClient(scripts={"a": SOLVE_SCRIPT, "b": SOLVE_SCRIPT})
    pol = parse_policy("alloy:a:0.5,b:0.5")

    async def go():
        return await run_episode(
            task, pol, mc, MockCTFEnv(task), max_steps=6, seed=7, race_id="r1"
        )

    t = asyncio.run(go())
    assert isinstance(t, Transcript)
    assert t.solved
    assert t.flags_found and t.flags_found[0].startswith("flag{")
    assistant_models = [m.model for m in t.messages if m.role == "assistant"]
    assert set(assistant_models) <= {"a", "b"}
    # per-turn model logging: model attribution present on every assistant turn
    assert all(m.model for m in t.messages if m.role == "assistant")
    assert t.steps >= 2
    assert t.tokens_in > 0 and t.tokens_out > 0
    assert getattr(t, "race_id", None) == "r1"


def test_scrub_flags() -> None:
    note = "hint: flag{abc-123-def-456-789-abc-123-def-456} is in flag.txt, try strings"
    scrubbed = scrub_flags(note)
    assert "flag{abc" not in scrubbed
    assert "[REDACTED]" in scrubbed
    assert "try strings" in scrubbed


def test_findings_bus_filters_flags() -> None:
    import asyncio

    bus = FindingsBus()
    a, b = bus.inbox_for("solo:a"), bus.inbox_for("solo:b")

    async def go():
        await bus.publish("solo:a", "flag{00000000-0000-0000-0000-000000000000}")
        await bus.publish("solo:a", "binary is stripped, try strings")

    asyncio.run(go())
    assert a.drain() == []  # sender does not receive its own note
    got = b.drain()
    assert got == ["flag{[REDACTED]}", "binary is stripped, try strings"]


def test_race_winner_transcripts_summary(tmp_path: Path) -> None:
    import asyncio

    task = make_task("race-task-1")
    fast = SoloPolicy(model="fast")
    slow = SoloPolicy(model="slow")
    never = SoloPolicy(model="never")
    mc = MockChatClient(
        scripts={
            "fast": [
                "```bash\nls\n```",
                "```bash\ncat flag.txt\n```",
                "done",
            ],
            "slow": [
                "```bash\nls -la\n```",
                "```bash\ncat README\n```",
                "```bash\nfile flag.txt\n```",
                "```bash\ncat flag.txt\n```",
                "done",
            ],
            "never": ["```bash\nls\n```"],  # last entry repeats forever
        }
    )

    async def go():
        return await race(
            task,
            [fast, slow, never],
            make_env_factory("mock"),
            mc,
            max_steps=10,
            out_dir=tmp_path,
            race_id="testrace",
        )

    result = asyncio.run(go())
    assert result.solved
    assert result.winner_policy == "solo:fast"
    by_policy = {t.policy: t for t in result.episodes}
    assert by_policy["solo:fast"].solved
    # slow + never were cancelled once the winner's verified flag landed
    assert not by_policy["solo:slow"].solved
    assert getattr(by_policy["solo:slow"], "cancelled", False) or by_policy["solo:slow"].steps < 10
    assert not by_policy["solo:never"].solved

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["winner"] == "solo:fast"
    assert summary["task_id"] == "race-task-1"
    assert set(summary["policies"]) == {"solo:fast", "solo:slow", "solo:never"}
    assert summary["policies"]["solo:fast"]["solved"] is True

    # transcripts on disk are valid §3.2 (validated via contracts models)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 3
    for fp in files:
        line = fp.read_text().strip()
        assert "\n" not in line
        t = Transcript.model_validate(json.loads(line))
        assert t.task_id == "race-task-1"
        assert "testrace" in t.episode_id
        # episodes that ran at least one step must carry per-turn model names
        if t.steps > 0:
            assert any(m.role == "assistant" and m.model for m in t.messages)
