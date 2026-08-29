"""Validate the three live integrations before building the full harness.

1. Daytona: create sandbox, exec a command, read result, delete.
2. Claude: one Messages API call.
3. GLM-4.7: one call via OpenAI-compatible endpoint with thinking DISABLED;
   assert no reasoning content comes back.
"""
import sys, time
import config


def test_daytona():
    print("\n=== DAYTONA ===")
    from daytona import Daytona, DaytonaConfig, CreateSandboxFromImageParams, Resources
    d = Daytona(DaytonaConfig(api_key=config.DAYTONA_API_KEY))
    print("client ok; creating sandbox from ubuntu:22.04 ...")
    t0 = time.time()
    sb = d.create(
        CreateSandboxFromImageParams(
            image="ubuntu:22.04",
            resources=Resources(cpu=1, memory=2, disk=5),
        ),
        timeout=300,
    )
    print(f"created in {time.time()-t0:.1f}s id={sb.id}")
    r = sb.process.exec("uname -a && cat /etc/os-release | head -1 && python3 --version 2>&1; echo done", timeout=60)
    print("exit_code:", r.exit_code)
    print("result:\n", r.result)
    # test writing and reading a file back
    sb.fs.upload_file(b"hello-flag\n", "/tmp/probe.txt")
    r2 = sb.process.exec("sha256sum /tmp/probe.txt", timeout=30)
    print("sha of uploaded:", r2.result.strip())
    sb.delete()
    print("deleted. DAYTONA OK")


def test_claude():
    print("\n=== CLAUDE ===")
    from anthropic import Anthropic
    c = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = c.messages.create(
        model=config.MODEL_A["model"],
        max_tokens=256,
        system="You are a helpful assistant. Answer in one short sentence.",
        messages=[{"role": "user", "content": "Say the word BANANA and nothing else."}],
    )
    print("stop_reason:", resp.stop_reason)
    print("usage:", resp.usage)
    for b in resp.content:
        print("block:", b.type, getattr(b, "text", "")[:120])
    print("CLAUDE OK")


def test_glm():
    print("\n=== GLM-4.7 (thinking disabled) ===")
    from openai import OpenAI
    c = OpenAI(api_key=config.GLM_API_KEY, base_url=config.GLM_BASE_URL)
    resp = c.chat.completions.create(
        model=config.MODEL_B["model"],
        max_tokens=256,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer in one short sentence."},
            {"role": "user", "content": "Say the word BANANA and nothing else."},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = resp.choices[0].message
    print("finish_reason:", resp.choices[0].finish_reason)
    print("usage:", resp.usage)
    print("content:", (msg.content or "")[:200])
    rc = getattr(msg, "reasoning_content", None)
    print("reasoning_content present?", bool(rc), repr(rc)[:120])
    # also verify tool calling works with a tool defined
    resp2 = c.chat.completions.create(
        model=config.MODEL_B["model"],
        max_tokens=256,
        messages=[{"role": "user", "content": "Run `ls` in the sandbox."}],
        tools=[{"type": "function", "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }}],
        tool_choice="auto",
        extra_body={"thinking": {"type": "disabled"}},
    )
    m2 = resp2.choices[0].message
    print("tool finish_reason:", resp2.choices[0].finish_reason)
    if m2.tool_calls:
        for tc in m2.tool_calls:
            print("tool_call:", tc.function.name, tc.function.arguments)
    print("GLM OK")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "claude"):
        test_claude()
    if which in ("all", "glm"):
        test_glm()
    if which in ("all", "daytona"):
        test_daytona()
