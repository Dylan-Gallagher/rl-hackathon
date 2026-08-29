"""Thin per-provider callers. Each takes the neutral Conversation, renders it,
calls the model with reasoning DISABLED, and returns (neutral_parts, usage).

Neither caller ever returns provider ids, stop reasons, thinking blocks or
signatures — only neutral text/tool_use parts (see conversation.parse_*).
"""
from __future__ import annotations
import time
import config
from conversation import Conversation, parse_anthropic_response, parse_openai_response

from anthropic import Anthropic
from openai import OpenAI

# Placeholder keys keep client construction (at import) from crashing when an
# env var is unset; a real call still fails clearly if the key is missing.
_anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY or "unset")
_glm = OpenAI(api_key=config.GLM_API_KEY or "unset", base_url=config.GLM_BASE_URL)
_qwen = OpenAI(api_key="EMPTY", base_url=config.QWEN_BASE_URL)

# Qwen3 non-thinking recommended sampling (Qwen team) — set explicitly so the
# baseline is not at the mercy of a serving default.
QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8)

MAX_OUTPUT_TOKENS = 4096


def _tools_anthropic(tools: list[dict]) -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]


def _tools_openai(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": t} for t in tools]


def _retry(fn, tries=5, base=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - broad on purpose for transient API errors
            last = e
            msg = str(e)
            # Non-retryable errors: surface immediately (don't burn backoff).
            if any(s in msg for s in ("invalid_request", "authentication", "permission",
                                       "Insufficient balance", "no resource package", "'1113'")):
                raise
            time.sleep(base * (2 ** i))
    raise last


def call_claude(conv: Conversation, tools: list[dict]) -> tuple[list[dict], dict]:
    system, messages = conv.to_anthropic()

    def _do():
        return _anthropic.messages.create(
            model=config.MODEL_A["model"],
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=messages,
            tools=_tools_anthropic(tools),
        )

    resp = _retry(_do)
    parts = parse_anthropic_response(resp)
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "model": config.MODEL_A["model"],
    }
    return parts, usage


def call_glm(conv: Conversation, tools: list[dict]) -> tuple[list[dict], dict]:
    messages = conv.to_openai()

    def _do():
        return _glm.chat.completions.create(
            model=config.MODEL_B["model"],
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=messages,
            tools=_tools_openai(tools),
            tool_choice="auto",
            extra_body={"thinking": {"type": "disabled"}},
        )

    resp = _retry(_do)
    parts = parse_openai_response(resp)
    # Safety net: assert the provider returned no reasoning content.
    rc = getattr(resp.choices[0].message, "reasoning_content", None)
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "model": config.MODEL_B["model"],
        "reasoning_content_leaked": bool(rc),
    }
    return parts, usage


def call_qwen(conv: Conversation, tools: list[dict]) -> tuple[list[dict], dict]:
    """Self-hosted Qwen3-8B via vLLM (OpenAI-compatible), thinking disabled.
    Same neutral-Conversation path as GLM, so the baseline is comparable."""
    messages = conv.to_openai()

    def _do():
        return _qwen.chat.completions.create(
            model=config.QWEN_MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=messages,
            tools=_tools_openai(tools),
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **QWEN_SAMPLING,
        )

    resp = _retry(_do)
    parts = parse_openai_response(resp)
    rc = getattr(resp.choices[0].message, "reasoning_content", None)
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "model": config.QWEN_MODEL,
        "reasoning_content_leaked": bool(rc),
    }
    return parts, usage


CALLERS = {"A": call_claude, "B": call_glm, "Q": call_qwen}
