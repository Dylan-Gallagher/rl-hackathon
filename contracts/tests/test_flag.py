"""Flag gen/verify/scan + capped_output tests."""

import re

from contracts.env.base import capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, new_flag, scan_for_flags, verify_flag
from contracts.task import Task


def test_new_flag_shape_unique():
    f1, f2 = new_flag(), new_flag()
    assert re.fullmatch(DEFAULT_FLAG_PATTERN, f1)
    assert f1 != f2


def test_new_flag_custom_template():
    t = Task(task_id="x", source="custom", category="misc",
             flag={"mode": "generated", "verify": "exact", "format": "CTF{uuid4}"})
    # '{uuid4}' hole is substituted with the bare uuid (standard formatting)
    assert new_flag(t).startswith("CTF") and new_flag(t) != new_flag(t)


def test_verify_exact_and_regex():
    assert verify_flag("flag{a}", "flag{a}", "exact")
    assert not verify_flag("flag{a}", "flag{b}", "exact")
    assert verify_flag("flag{a}", "flag{a}", "regex", r"flag\{a\}")
    assert not verify_flag("flag{X}", "flag{a}", "regex", r"flag\{[aX]\}")


def test_scan_for_flags_default_and_custom():
    text = "out flag{%s} more flag{%s}" % ("a" * 36, "a" * 36)
    assert scan_for_flags(text) == [f"flag{{{'a' * 36}}}"]
    assert scan_for_flags("picoCTF{hi}", r"picoCTF\{[^}]*\}") == ["picoCTF{hi}"]
    assert scan_for_flags("nothing here") == []


def test_capped_output_head_tail():
    assert capped_output("short", 100) == "short"
    text = "H" * 600 + "MID" + "T" * 400
    out = capped_output(text, 100)
    assert out.startswith("H" * 60)
    assert out.endswith("T" * 40)
    assert "...[truncated 903 chars]..." in out
