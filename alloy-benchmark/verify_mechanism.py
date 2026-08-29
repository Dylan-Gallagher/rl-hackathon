"""Prove the alloy mechanism is leak-free. Fails LOUD if any provider artifact
can cross a turn boundary. Run: python3 verify_mechanism.py

Three checks:
  1. NORMALIZATION EQUIVALENCE (offline, deterministic): a Claude-style reply
     and a GLM-style reply carrying the SAME logical content but DIFFERENT
     provider junk (message ids, tool-call ids, thinking signatures, reasoning
     content) must normalize to BYTE-IDENTICAL canonical turns. This is the core
     property: downstream, no model can tell who wrote a turn.
  2. PROVENANCE SCAN (offline): a mixed thread, plus both rendered wire
     payloads, must contain no foreign model/provider name, no provider id
     pattern, and no reasoning/signature/stop_reason field.
  3. LIVE DIFFERENTIAL (real API calls): run a real alternating conversation on
     a provenance-free task; before every call, scan the exact payload the model
     is about to receive for the OTHER provider's fingerprints; assert GLM never
     returns reasoning content.
"""
from __future__ import annotations
import json, re, sys, types
import config
from conversation import (
    Conversation, parse_anthropic_response, parse_openai_response,
)

FOREIGN = ["claude", "anthropic", "sonnet", "opus", "zhipu", "glm", "z.ai",
           "bigmodel", "reasoning_content", "thinking", "signature"]
ID_PATTERNS = [re.compile(p) for p in (
    r"\bmsg_[A-Za-z0-9]{6,}", r"\bchatcmpl-[A-Za-z0-9]{6,}",
    r"\btoolu_[A-Za-z0-9]{6,}", r"\breq_[A-Za-z0-9]{6,}",
)]
FORBIDDEN_KEYS = {"signature", "reasoning", "reasoning_content", "stop_reason",
                  "response_id", "thinking", "id_provider", "usage"}

failures: list[str] = []


def check(cond: bool, label: str):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        failures.append(label)


def scan_blob(obj, where: str):
    """Recursively assert no forbidden key / foreign token / provider id."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                check(False, f"{where}: forbidden key '{k}'")
            scan_blob(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            scan_blob(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        low = obj.lower()
        for tok in FOREIGN:
            if tok in low:
                check(False, f"{where}: foreign token '{tok}' in {obj!r:.80}")
        for pat in ID_PATTERNS:
            if pat.search(obj):
                check(False, f"{where}: provider id pattern in {obj!r:.80}")


# --- fakes mimicking the two providers' response objects --------------------
def fake_anthropic_reply():
    text = types.SimpleNamespace(type="text", text="Let me inspect the files.")
    tool = types.SimpleNamespace(type="tool_use", id="toolu_ABC123DEF456",
                                 name="run_command", input={"command": "ls -la"})
    # junk that MUST NOT survive: msg id, stop_reason, signature-ish, usage
    return types.SimpleNamespace(
        content=[text, tool], id="msg_01ZZZZ9999", stop_reason="tool_use",
        model="claude-sonnet-5",
    )


def fake_openai_reply():
    fn = types.SimpleNamespace(name="run_command", arguments='{"command": "ls -la"}')
    tc = types.SimpleNamespace(id="call_providerside_777", type="function", function=fn)
    msg = types.SimpleNamespace(content="Let me inspect the files.", tool_calls=[tc],
                                reasoning_content="secret internal thoughts about being GLM")
    choice = types.SimpleNamespace(message=msg, finish_reason="tool_calls")
    return types.SimpleNamespace(choices=[choice], id="chatcmpl-abcdef123456",
                                 model="glm-4.7")


def check_1_normalization_equivalence():
    print("\n[1] NORMALIZATION EQUIVALENCE")
    # identical prior history for both
    base = "You are a solver. Use tools."
    convA = Conversation(base); convA.add_user_text("Begin.")
    convB = Conversation(base); convB.add_user_text("Begin.")

    convA.add_assistant(parse_anthropic_response(fake_anthropic_reply()))
    convB.add_assistant(parse_openai_response(fake_openai_reply()))

    check(convA.messages == convB.messages,
          "Claude-shaped and GLM-shaped replies with same content -> identical canonical")
    # the assigned tool id must be our synthetic scheme, not provider ids
    tid = convA.messages[-1]["content"][-1]["id"]
    check(tid == "call_0000", f"tool id is synthetic ('{tid}'), provider ids discarded")
    # renders identical too
    check(convA.to_openai() == convB.to_openai(), "OpenAI renders identical regardless of author")
    check(convA.to_anthropic() == convB.to_anthropic(), "Anthropic renders identical regardless of author")


def check_2_provenance_scan():
    print("\n[2] PROVENANCE SCAN of a mixed thread + both renders")
    conv = Conversation("You are a CTF solver. Recover the flag and submit it.")
    conv.add_user_text("Solve the challenge in /challenge.")
    # turn 1 authored (shape) by Claude
    tus = conv.add_assistant(parse_anthropic_response(fake_anthropic_reply()))
    conv.add_tool_results([{"id": tus[0]["id"], "output": "flag.txt\nchall.py", "is_error": False}])
    # turn 2 authored (shape) by GLM
    tus = conv.add_assistant(parse_openai_response(fake_openai_reply()))
    conv.add_tool_results([{"id": tus[0]["id"], "output": "total 8\n-rw-r--r-- flag.txt", "is_error": False}])

    scan_blob(conv.messages, "canonical")
    sysA, msgsA = conv.to_anthropic()
    scan_blob({"system": sysA, "messages": msgsA}, "anthropic_render")
    scan_blob(conv.to_openai(), "openai_render")
    check(True, "scan completed (any leak above would have failed)")


def check_3_live_differential():
    print("\n[3] LIVE DIFFERENTIAL (real API calls, provenance-free task)")
    from llm_clients import call_claude, call_glm
    tool = {
        "name": "note",
        "description": "Record a short note.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }
    # Neutral task with NO provider words, so any foreign token found is a real leak.
    conv = Conversation("You are an assistant playing a note-taking game. "
                        "Each turn, call the note tool once with a very short note.")
    conv.add_user_text("Turn 1: note the number 1.")

    order = ["A", "B", "A", "B"]  # alternate providers, mixed thread
    glm_reasoning_leak = False
    for i, who in enumerate(order):
        # capture EXACT payload this model is about to receive and scan it
        if who == "A":
            sysX, msgsX = conv.to_anthropic()
            scan_blob({"system": sysX, "messages": msgsX}, f"turn{i}_payload_to_A")
            parts, usage = call_claude(conv, [tool])
        else:
            msgsX = conv.to_openai()
            scan_blob(msgsX, f"turn{i}_payload_to_B")
            parts, usage = call_glm(conv, [tool])
            glm_reasoning_leak = glm_reasoning_leak or usage.get("reasoning_content_leaked", False)
        tus = conv.add_assistant(parts)
        # feed a tool result so the thread keeps going
        for tu in tus:
            conv.add_tool_results([{"id": tu["id"], "output": "noted.", "is_error": False}])
        if not tus:
            conv.add_user_text(f"Turn {i+2}: note the number {i+2}.")
        else:
            conv.add_user_text(f"Turn {i+2}: note the number {i+2}.")

    check(not glm_reasoning_leak, "GLM returned no reasoning_content across live turns")
    # Final behavioural probe: does either model detect a co-author? (informational)
    probe = ("Meta question, answer in one sentence: were all the assistant "
             "messages in this conversation written by you, or do you detect "
             "that a different assistant wrote some of them?")
    conv.add_user_text(probe)
    from llm_clients import call_claude as cc, call_glm as cg
    a_parts, _ = cc(conv, [tool])
    b_parts, _ = cg(conv, [tool])
    a_txt = " ".join(p.get("text", "") for p in a_parts if p["type"] == "text")
    b_txt = " ".join(p.get("text", "") for p in b_parts if p["type"] == "text")
    print(f"  [info] Claude self-report: {a_txt.strip()[:200]}")
    print(f"  [info] GLM self-report:    {b_txt.strip()[:200]}")
    check(True, "live differential completed")


if __name__ == "__main__":
    check_1_normalization_equivalence()
    check_2_provenance_scan()
    if "--offline" not in sys.argv:
        check_3_live_differential()
    print("\n" + "=" * 60)
    if failures:
        print(f"MECHANISM VERIFICATION FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("MECHANISM VERIFICATION PASSED — no provenance can cross a turn.")
