"""Model, dataset, and OpenAI-compatible backend helpers.

This file is derived from HackSynth-GRPO commit
98128055275c001eb7c69005795f323298bf79e9 and modified for the Path 1
reproduction. See UPSTREAM.md and LICENSE.md.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace


class DummyGen:
    """Mimic an Unsloth/vLLM generation object."""

    def __init__(self, txt: str):
        self.outputs = [SimpleNamespace(text=txt)]


class DummyTokenizer:
    """Small tokenizer adapter used by OpenAI-compatible evaluation clients."""

    def __init__(self, enc):
        self._enc = enc
        self._special = {
            "<|start_header_id|>": 32000,
            "<|end_header_id|>": 32001,
            "<|eot_id|>": 32002,
            "assistant": 32003,
        }

    def apply_chat_template(self, messages, tools=None, tokenize=False, **kwargs):
        if isinstance(messages, dict):
            messages = messages["messages"]
        return "".join(
            f"<|start_header_id|>{message['role']}<|end_header_id|>{message['content']}"
            for message in messages
        )

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        ids = self._enc.encode(text)
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor(ids).unsqueeze(0)}
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self._enc.encode(token)[0])


def get_model(
    model_id: str,
    lora_path: str | None = None,
    *,
    for_training: bool = False,
    max_seq_length: int = 8192,
    lora_rank: int = 32,
    gpu_memory_utilization: float = 0.6,
    random_state: int = 3407,
):
    """Load the supported Llama model, optionally adding/loading a LoRA.

    Imports Unsloth lazily so CPU-only audit and report commands do not require
    CUDA packages. A fresh LoRA is attached only for training; baseline
    evaluation uses the unmodified base model.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path or model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    if for_training and not lora_path:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=random_state,
        )
    return model, tokenizer


def load_data(
    data_path: str,
    include_hint: bool,
    difficulties: list[str],
    *,
    seed: int = 42,
    limit: int | None = None,
):
    """Load Random-Crypto data with explicit ``all`` difficulty semantics."""
    import pandas as pd
    from datasets import Dataset

    frame = pd.read_csv(data_path)
    required = {"question", "necessary_info", "hint", "flag", "difficulty"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{data_path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["source_row"] = range(len(frame))
    prefix = "random-crypto-verified" if len(frame) == 50 else "random-crypto-train"
    frame["task_id"] = frame["source_row"].map(lambda index: f"{prefix}-{index:04d}")

    requested = set(difficulties)
    if not requested:
        raise ValueError("At least one difficulty or 'all' is required")
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("'all' cannot be combined with named difficulties")
    else:
        frame = frame[frame["difficulty"].isin(requested)]

    print(f"Loaded {len(pd.read_csv(data_path))} rows from {data_path}")
    print(f"Keeping difficulties {difficulties}: {len(frame)} rows")

    frame = frame.rename(columns={"flag": "answer"})
    frame["question"] = frame["question"].fillna("").str.cat(
        frame["necessary_info"].fillna(""), sep="\n\n"
    )
    if include_hint:
        frame["question"] = frame["question"].str.cat(
            frame["hint"].fillna(""), sep="\nHint: "
        )

    dataset = Dataset.from_pandas(frame, preserve_index=False).rename_column("question", "prompt")
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _get_tokenizer(model_id: str):
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return DummyTokenizer(encoding)


def create_llm_pipeline(backend: str, model_id: str):
    from openai import OpenAI

    if backend == "vllm":
        client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", "sk-fake"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        )
    elif backend == "openai":
        client = OpenAI()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return client, _get_tokenizer(model_id)


def generate_text(model, llm_pipeline, messages, temperature, top_p, max_new_tokens, n):
    from openai import BadRequestError

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "n": n,
        }
        if model == "o3":
            kwargs["max_completion_tokens"] = max_new_tokens
        else:
            kwargs.update(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        return llm_pipeline.chat.completions.create(**kwargs)
    except BadRequestError as error:
        body = error.body
        if isinstance(body, str):
            import json

            try:
                body = json.loads(body)
            except Exception:
                raise error
        if (
            error.status_code == 400
            and isinstance(body, dict)
            and body.get("error", {}).get("code") == "invalid_prompt"
        ):
            raise RuntimeError("POLICY_BLOCKED") from error
        raise
