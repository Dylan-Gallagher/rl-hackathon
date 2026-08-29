"""Evaluate a tool-use agent on Random-Crypto.

Derived from HackSynth-GRPO commit
98128055275c001eb7c69005795f323298bf79e9 and modified for reproducible,
machine-readable Path 1 evaluation. See UPSTREAM.md and LICENSE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

from trl.trainer.grpo_trainer import apply_chat_template
from vllm import SamplingParams

import rl_helpers
from helpers import DummyGen, create_llm_pipeline, generate_text, get_model, load_data
from rl_helpers import (
    AgenticOutputs,
    _last_assistant_text,
    check_exceeded_max_new_tokens,
    check_has_answered_question,
    get_initial_chat,
    run_tool_calls,
)

args = None


def _stringify_messages(messages):
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False)
        if message.get("role") == "tool":
            normalized.append({"role": "user", "content": f"[TOOL OUTPUT]\n{content}"})
        else:
            normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


def _generate_agent_turn(generate_fn, tokenizer, chat_states):
    requests = []
    indices = []
    for index, state in enumerate(chat_states):
        if state.get("finished"):
            continue
        if state["messages"][-1]["role"] in {"tool", "system", "user"}:
            rendered = apply_chat_template(state, tokenizer=tokenizer)["text"]
            requests.append({"messages": state["messages"], "rendered_prompt": rendered})
            indices.append(index)

    if not requests:
        return chat_states

    responses = generate_fn(requests)
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    for response, index in zip(responses, indices):
        full_text = response.outputs[0].text
        assistant_text = full_text.split(marker)[-1]
        chat_states[index]["messages"].append({"role": "assistant", "content": assistant_text})
        if full_text.startswith("[POLICY_BLOCKED]"):
            chat_states[index]["finished"] = True
    return chat_states


def run_agent_eval(generate_fn, tokenizer, questions, max_generations=4, max_new_tokens=8192):
    chat_states = [get_initial_chat(question) for question in questions]
    for step in range(max_generations):
        print(f"Agentic generation step: {step}")
        chat_states = _generate_agent_turn(generate_fn, tokenizer, chat_states)
        chat_states = check_has_answered_question(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(
            chat_states, max_new_tokens, tokenizer=tokenizer
        )

    import mcp

    mcp.mcp_call_tool(
        "execute_python", {"code": "print('Resetting MCP server')", "reset": True}
    )
    final = [_last_assistant_text(state) for state in chat_states]
    return AgenticOutputs([], [], [], final, chat_states)


def eval_generate_fn(batch_requests):
    outputs = []
    for request in batch_requests:
        rendered_prompt = request.get("rendered_prompt", "")
        messages = request.get("messages", [])
        if args.backend == "local":
            outputs.append(
                model.fast_generate(
                    [rendered_prompt],
                    sampling_params=SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.generation_max_tokens,
                        seed=args.seed,
                    ),
                )[0]
            )
            continue

        try:
            response = generate_text(
                model=args.model_id,
                llm_pipeline=model,
                messages=_stringify_messages(messages),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.generation_max_tokens,
                n=1,
            )
            outputs.append(DummyGen(response.choices[0].message.content or ""))
        except RuntimeError as error:
            if str(error) != "POLICY_BLOCKED":
                raise
            outputs.append(DummyGen("[POLICY_BLOCKED]"))
        except Exception as error:
            # Keep the episode in the denominator and make failures inspectable.
            print(f"[Generation error] {type(error).__name__}: {error}")
            outputs.append(DummyGen(f"[GENERATION_ERROR] {type(error).__name__}: {error}"))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Path 1 tool-use agent")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--backend", choices=["local", "vllm", "openai"], required=True)
    parser.add_argument("--include_hint", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["all"],
        choices=["all", "easy", "medium", "hard", "not-defined"],
    )
    parser.add_argument("--lora_path", default=None, help="Final LoRA adapter for local evaluation")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Whole trajectory token cap")
    parser.add_argument("--generation_max_tokens", type=int, default=2048, help="Per-turn token cap")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic task limit")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parsed = parser.parse_args()
    if parsed.k < 1 or parsed.max_generations < 1 or parsed.max_new_tokens < 1:
        parser.error("k, max_generations, and max_new_tokens must be positive")
    if parsed.limit is not None and parsed.limit < 1:
        parser.error("limit must be positive")
    if parsed.lora_path and parsed.backend != "local":
        parser.error("--lora_path is supported only by --backend local; serve it explicitly for vLLM")
    parsed.difficulties = [
        "not defined" if difficulty == "not-defined" else difficulty
        for difficulty in parsed.difficulties
    ]
    return parsed


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.run_id = args.run_id or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    config_record = vars(args).copy()
    config_record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "run_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.backend == "local":
        model, tokenizer = get_model(
            args.model_id,
            lora_path=args.lora_path,
            for_training=False,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        model.run_agent = rl_helpers.run_agent
    else:
        model, tokenizer = create_llm_pipeline(args.backend, args.model_id)
        model.run_agent = run_agent_eval

    test_dataset = load_data(
        args.data_path,
        args.include_hint,
        difficulties=args.difficulties,
        seed=args.seed,
        limit=args.limit,
    )
    rl_helpers.run_eval_pass_majority(
        model=model,
        test_dataset=test_dataset,
        generate_fn=eval_generate_fn,
        tokenizer=tokenizer,
        k=args.k,
        max_generations=args.max_generations,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        run_id=args.run_id,
        model_id=args.model_id,
        backend=args.backend,
    )
