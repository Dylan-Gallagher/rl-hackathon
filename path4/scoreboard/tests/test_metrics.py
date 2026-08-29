from __future__ import annotations

from contracts import Transcript, TranscriptMessage, episode_id_for, write_transcript
from path4.scoreboard.metrics import aggregate

# ---------------------------------------------------------------------------
# Fixture: 2 policies x 3 tasks, k=4 episodes where available.
# Outcomes are hand-picked so Pass@k / Maj@k can be computed by hand.
#
# Policy A "solo:test":
#   t1: SSSS   -> pass@4 = 1, maj@4 = 1 ; pass@2 = 1, maj@2 = 1
#   t2: FFFF   -> pass@4 = 0, maj@4 = 0 ; pass@2 = 0, maj@2 = 0
#   t3: SF (only 2 episodes, < k=4) -> excluded from @4; pass@2 = 1, maj@2 = 0
#   Pass@1 = 5 solves / 10 episodes = 0.5
#   Pass@2 = (1 + 0 + 1)/3 = 2/3 ; Maj@2 = (1+0+0)/3 = 1/3
#   Pass@4 = (1+0)/2 = 0.5 ; Maj@4 = (1+0)/2 = 0.5
#
# Policy B "alloy:x,y" (episodes ordered by episode_id):
#   t1: FSFS -> pass@4 = 1, maj@4 = 0 ; pass@2 = 1 - 1/6 = 5/6, maj@2 = 0
#   t2: SSFF -> pass@4 = 1, maj@4 = 0 ; pass@2 = 5/6, maj@2 = 1
#   t3: SSSF -> pass@4 = 1, maj@4 = 1 ; pass@2 = 1, maj@2 = 1
#   Pass@1 = 7/12 ; Pass@2 = (5/6+5/6+1)/3 = 8/9 ; Maj@2 = 2/3 ; Maj@4 = 1/3
# ---------------------------------------------------------------------------

CATS = {"t1": "pwn", "t2": "crypto", "t3": "web"}
OUTCOMES = {
    "solo:test": {"t1": [1, 1, 1, 1], "t2": [0, 0, 0, 0], "t3": [1, 0]},
    "alloy:x,y": {"t1": [0, 1, 0, 1], "t2": [1, 1, 0, 0], "t3": [1, 1, 1, 0]},
}


def make_fixture() -> list[Transcript]:
    ts = []
    for policy, tasks in OUTCOMES.items():
        for task, outs in tasks.items():
            for idx, solved in enumerate(outs):
                ts.append(
                    Transcript(
                        task_id=task,
                        episode_id=episode_id_for(policy, task, idx),
                        policy=policy,
                        split="eval",
                        messages=[
                            TranscriptMessage(turn=0, role="assistant", content="hi", model="m"),
                            TranscriptMessage(turn=0, role="tool", content="out"),
                        ],
                        solved=bool(solved),
                        steps=10 * (idx + 1) if solved else 20,
                        flags_found=["flag{x}"] if solved else [],
                        tokens_in=100,
                        tokens_out=50 if solved else 500,
                        category=CATS[task],
                    )
                )
    return ts


def approx(a, b, eps=1e-6):
    return a is not None and abs(a - b) < eps


def test_pass1_and_episode_counts():
    s = aggregate(make_fixture(), ks=(1, 2, 4))
    pol = {p["policy"]: p for p in s["policies"]}
    assert s["episodes"] == 22
    a, b = pol["solo:test"], pol["alloy:x,y"]
    assert a["episodes"] == 10 and a["solves"] == 5 and approx(a["solve_rate"], 0.5)
    assert b["episodes"] == 12 and b["solves"] == 7 and approx(b["solve_rate"], 7 / 12)
    assert approx(a["pass_at_k"]["1"], 0.5)
    assert approx(b["pass_at_k"]["1"], 7 / 12)


def test_pass_k_unbiased_hand_computed():
    pol = {p["policy"]: p for p in aggregate(make_fixture(), ks=(2, 4))["policies"]}
    a, b = pol["solo:test"], pol["alloy:x,y"]
    # tasks with <4 episodes (t3 for A) are excluded from @4, not counted at all
    assert approx(a["pass_at_k"]["4"], 0.5)  # (t1:1 + t2:0) / 2 tasks
    assert approx(b["pass_at_k"]["4"], 1.0)  # every task has a solve in all 4 draws
    assert approx(a["pass_at_k"]["2"], 2 / 3)  # (1 + 0 + 1)/3
    assert approx(b["pass_at_k"]["2"], 8 / 9)  # (5/6 + 5/6 + 1)/3


def test_maj_k_first_k_ordering_and_tie_is_not_solved():
    pol = {p["policy"]: p for p in aggregate(make_fixture(), ks=(2, 4))["policies"]}
    a, b = pol["solo:test"], pol["alloy:x,y"]
    # A: t1 SS->1, t2 FF->0, t3 first-2 = SF -> tie(1 of 2) NOT solved -> 0
    assert approx(a["maj_at_k"]["2"], 1 / 3)
    assert approx(a["maj_at_k"]["4"], 0.5)
    # B: t1 FSFS -> 2/4 tie not > k/2 -> 0 ; t2 SSFF -> 0 ; t3 SSSF -> 1
    assert approx(b["maj_at_k"]["4"], 1 / 3)
    assert approx(b["maj_at_k"]["2"], 2 / 3)  # (0 + 1 + 1)/3


def test_k_greater_than_n_is_guarded():
    # single task, 2 episodes, k=8 -> no @8 estimate at all
    ts = [
        Transcript(task_id="only", episode_id="e0", policy="solo:g", solved=True),
        Transcript(task_id="only", episode_id="e1", policy="solo:g", solved=False),
    ]
    s = aggregate(ts, ks=(1, 8))
    p = s["policies"][0]
    assert p["pass_at_k"]["8"] is None
    assert p["maj_at_k"]["8"] is None
    assert approx(p["pass_at_k"]["1"], 0.5)


def test_category_fallback_and_breakdown():
    ts = [
        Transcript(task_id="t", episode_id="e0", policy="solo:g", solved=True),  # no category
        Transcript(task_id="t", episode_id="e1", policy="solo:g", solved=False, category="pwn"),
    ]
    p = aggregate(ts)["policies"][0]
    assert set(p["categories"]) == {"pwn", "unknown"}
    assert p["categories"]["unknown"]["solves"] == 1
    assert p["categories"]["pwn"]["episodes"] == 1


def test_steps_tokens_solved_vs_unsolved():
    p = {x["policy"]: x for x in aggregate(make_fixture(), ks=(1,))["policies"]}["solo:test"]
    # A solved steps: t1 [10,20,30,40] + t3 [10] -> avg 22 ; unsolved: t2 [20x4] + t3 [20] -> 20
    assert p["avg_steps_solved"] == 22.0
    assert p["avg_steps_unsolved"] == 20.0
    # tokens: solved 150 each, unsolved 600 each
    assert p["avg_tokens_solved"] == 150.0
    assert p["avg_tokens_unsolved"] == 600.0


def test_race_wins():
    races = [
        {"race_id": "r1", "task_id": "t1", "winner": "solo:test"},
        {"race_id": "r2", "task_id": "t2", "winner": "solo:test"},
        {"race_id": "r3", "task_id": "t3", "winner": "alloy:x,y"},
    ]
    pol = {p["policy"]: p for p in aggregate(make_fixture(), races=races)["policies"]}
    assert pol["solo:test"]["race_wins"] == 2
    assert pol["alloy:x,y"]["race_wins"] == 1


def test_deterministic_ordering():
    s = aggregate(make_fixture())
    assert [p["policy"] for p in s["policies"]] == sorted(p["policy"] for p in s["policies"])
    for p in s["policies"]:
        assert list(p["categories"]) == sorted(p["categories"])


def test_roundtrip_through_disk(tmp_path):
    from path4.scoreboard.metrics import scan_race_summaries, scan_transcripts

    ts = make_fixture()
    for t in ts:
        write_transcript(t, tmp_path / f"{t.policy.replace(':', '_')}.jsonl")
    (tmp_path / "runs" / "race-1").mkdir(parents=True)
    (tmp_path / "runs" / "race-1" / "summary.json").write_text(
        '{"race_id": "r1", "task_id": "t1", "winner": "solo:test"}'
    )
    # a summary.json without the race keys must be ignored
    (tmp_path / "runs" / "race-2").mkdir(parents=True)
    (tmp_path / "runs" / "race-2" / "summary.json").write_text('{"other": true}')

    got = aggregate(scan_transcripts(tmp_path), scan_race_summaries(tmp_path), ks=(2, 4))
    pol = {p["policy"]: p for p in got["policies"]}
    assert got["episodes"] == 22
    assert approx(pol["solo:test"]["pass_at_k"]["4"], 0.5)
    assert pol["solo:test"]["race_wins"] == 1
    assert got["races"] == 1
