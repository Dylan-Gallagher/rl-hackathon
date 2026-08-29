from __future__ import annotations

from path4.verl_grpo.reward import GroupStats, dapo_filter, flag_reward, reward_group


def test_flag_reward_binary():
    assert flag_reward(True, ["flag{abc}"]) == 1.0
    assert flag_reward(True, []) == 1.0      # solved is the verifier's word
    assert flag_reward(False, []) == 0.0
    assert flag_reward(False, ["flag{abc}"]) == 0.0  # found flag but unverified solve -> 0


def test_reward_group_stats():
    s = reward_group([0.0, 0.0])
    assert s.all_zero and not s.all_one and s.mean == 0.0 and s.variance == 0.0
    s = reward_group([1.0, 1.0])
    assert s.all_one and not s.all_zero
    s = reward_group([0.0, 1.0])
    assert not s.all_zero and not s.all_one
    assert s.mean == 0.5 and s.variance == 0.25 and abs(s.std - 0.5) < 1e-9
    s = reward_group([1.0, 0.0, 1.0, 1.0])
    assert s.mean == 0.75 and abs(s.variance - 0.1875) < 1e-9


def test_reward_group_empty_safe():
    s = reward_group([])
    assert s == GroupStats(0.0, 0.0, True, False, 0.0)


def test_dapo_filter():
    groups = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0, 0.0, 1.0], []]
    assert dapo_filter(groups) == [False, False, True, True, False]
