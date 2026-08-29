"""
RL helpers module for handling tool-based conversations.
This module provides utility functions for handling chat-based tool interactions
and calculating rewards based on the quality of responses.
"""

import json
import re
import asyncio
import torch
from datetime import datetime
from dataclasses import dataclass
import nest_asyncio
nest_asyncio.apply()
from typing import List, Callable
import re
import mcp
import os
import time
from pathlib import Path
from trl.trainer.grpo_trainer import apply_chat_template
from path1_tools.metrics import compute_eval_metrics, extract_flag, extract_flags

LOG_DIR = f"logs/{time.time()}"

MCP_TOOLS = mcp.list_tools_sync()
MCP_TOOL_LIST = json.dumps(MCP_TOOLS, indent=2)
MCP_TOOL_NAMES = {tool["name"] for tool in MCP_TOOLS}

FLAG_BOX_RE = re.compile(r"\\boxed\{(?:flag|picoCTF)\{[^}]+\}\}", flags=re.IGNORECASE)

# Constants for prompts and tool definitions
def get_system_prompt():
    """Get the system prompt with current date."""
    return f"""
You are a helpful assistant with tool calling capabilities.
"""


def build_user_prompt(q):
    return f"""
You are an assistant with access to the tools listed below.

### Rules (follow **all** of them) ###
1. Do your chain-of-thought inside **<reasoning> … </reasoning>** tags.
2. **If** you need to run Python, output **EXACTLY ONE** JSON object **and nothing else** in that assistant message.  
   The object *must* have the shape

        {{
          "name": "execute_python",
          "inputs": {{
            "code": "print('hello')",
            "reset": false
          }}
        }}

   – no prose before or after, no Markdown formatting.
3. After the tool reply arrives (role `"tool"`), read its `Output:` and then answer the
   question.  Wrap the final flag in **\\boxed{{flag{{…}}}}**.  
   **Never** guess the flag before you have seen it in a tool response.
4. **Never** include \boxed{{flag{{…}}}} in the SAME assistant message that
   contains a JSON tool call.  Only write the boxed flag **after** the
   tool’s reply (role="tool") has appeared.
5. Every assistant message must obey these rules.

### Available tools ###
{MCP_TOOL_LIST}

Question: {q}
"""


def get_initial_chat(question):
    """
    Initialize a chat state with the question.
    
    Args:
        question (str): The question to ask
        
    Returns:
        dict: Initial chat state with system and user messages
    """
    return {"messages":[
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": build_user_prompt(question)},
    ]}


def extract_json_objects(text: str):
    """
    Return every top-level JSON object found in *text*.
    Safe against braces inside strings because it lets the real
    JSON parser (`raw_decode`) do the heavy lifting.
    """
    decoder = json.JSONDecoder()
    idx = 0
    found = []

    while True:
        try:
            idx = text.index('{', idx)          # next candidate
        except ValueError:
            break                               # no more '{' → done

        try:
            obj, end = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                found.append(obj)
            idx += end                          # jump past this object
        except json.JSONDecodeError:
            idx += 1                            # not valid here → try one char later
    return found



def run_agent_generations(generate_fn, tokenizer, chat_states):
    """
    Run generation for chat states requiring assistant responses.
    
    Args:
        generate_fn: Function to generate responses
        tokenizer: Tokenizer for processing text
        chat_states: List of chat states
        
    Returns:
        list: Updated chat states
    """
    prompts = []
    batch_indices = []
    # Prepare prompts for chat states needing an assistant response.
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            continue

        if chat_state["messages"][-1]["role"] in ["tool", "system", "user"]:
                prompt = apply_chat_template(chat_state, tokenizer=tokenizer)['text']
                prompts.append(prompt)
                batch_indices.append(idx)

    if prompts:
        responses = generate_fn(prompts)
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            full_response = responses[i].outputs[0].text
            assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            chat_state["messages"].append({
                "role": "assistant",
                "content": assistant_response
            })
    return chat_states


def run_tool_calls(chat_states):
    """
    Execute tool calls found in chat states.
    
    Args:
        chat_states: List of chat states
        
    Returns:
        list: Updated chat states with tool call results
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant to run tool calls"
        try:
            assistant_msg = chat_state["messages"][-1]["content"]
            calls = extract_json_objects(assistant_msg)
            dispatched = False
            if len(calls) >= 1:
                for i in range(len(calls)):
                    if "name" in calls[i] and calls[i]["name"] in MCP_TOOL_NAMES:
                        if "reset" in calls[i]["inputs"]:
                            calls[i]["inputs"]["reset"] = False
                        output = mcp.mcp_call_tool(calls[i]["name"], calls[i]["inputs"])
                        output["role"] = "tool"
                        chat_state["messages"].append(output)
                        dispatched = True
                        break
                if not dispatched:
                    chat_state["messages"].append({
                        "role": "system",
                        "content": f"Tool name was not found in the tool call, or the JSON syntax was wrong)"
                    })
            else:
                chat_state["messages"].append({
                    "role": "system",
                    "content": f"No tool calls found with the JSON parser."
                })
        except Exception as e:
            chat_state["messages"].append({
                "role": "system",
                "content": f"Error during post-processing: {str(e)}"
            })
            chat_state["finished"] = True
    return chat_states

def get_mask(text, tokenizer):
    encoding = tokenizer(text, add_special_tokens=False)
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_ranges = []
    i = 0
    while i < len(encoding.input_ids) - 1:
        if encoding.input_ids[i] == start_header_id and encoding.input_ids[i+1] == assistant_token:
            i += 2
            while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
                i += 1
            i += 2
            start_idx = i
            while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                i += 1
            end_idx = i
            assistant_ranges.append((start_idx, end_idx))
        else:
            i += 1
    mask = [0] * len(encoding.input_ids)
    for start_idx, end_idx in assistant_ranges:
        for idx in range(start_idx, end_idx):
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.int)

def check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        new_length = get_chat_num_tokens(chat_state, tokenizer)
        if new_length > max_new_tokens:
            while get_chat_num_tokens(chat_state, tokenizer) > max_new_tokens-50: # account for the added system message
                chat_state["messages"].pop(-1) # get rid of too long parts -> can sometimes throw error during training
            chat_state["messages"].append({
                "role": "system",
                "content": f"Exceeded max_new_tokens ({new_length}/{max_new_tokens})"
            })
            chat_state["finished"] = True
    return chat_states

@dataclass
class AgenticOutputs:
    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


def get_chat_num_tokens(chat_state, tokenizer):
    chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)['text']
    return tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze().shape[0]


def _last_assistant_text(chat):
    for msg in reversed(chat["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return "" 


def check_has_answered_question(chat_states):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue

        last_msg = _last_assistant_text(chat_state)
        if FLAG_BOX_RE.search(last_msg):
            chat_state["messages"].append({
                "role": "system",
                "content": """Found answer with \boxed{flag{}} syntax"""
            })
            chat_state["finished"] = True
    return chat_states

def run_agent(generate_fn, tokenizer, questions, max_generations=5, max_new_tokens=8192):
    """
    Run the agent to completion for a batch of questions.
    
    Args:
        generate_fn: Function to generate model responses
        tokenizer: Tokenizer for processing text
        batch: Batch of data containing questions
        max_generations: Maximum number of generation steps
        
    Returns:
        list: Final answers for each question
    """
    chat_states = [get_initial_chat(q) for q in questions]
    # set the initial_prompt length
    for chat_state in chat_states:
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)

    # agent loop
    for i in range(max_generations):
        print("Agentic generation step:", i)
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_has_answered_question(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer)
    
    print("Reseting MCP server variables...")
    mcp.mcp_call_tool("execute_python", {"code": "print('Reseting MCP server')",  "reset": True})

        
    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])

    def split_prompt_assistant(convo_text):
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            raise ValueError(f"Could not find assistant marker in conversation text. Conversation that caused this: {convo_text}")
            return convo_text, ""
        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[:idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker):]
        return prompt, assistant_response
    
    str_chats = [apply_chat_template(chat, tokenizer=tokenizer)['text'] for chat in chat_states]
    prompt_toks, response_toks, response_masks = [], [], []
    for str_chat in str_chats:
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze())
        response_toks.append(tokenizer(response, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze()[:max_new_tokens])
        mask = get_mask(str_chat, tokenizer)[len(prompt_toks[-1]):][:max_new_tokens]

        response_masks.append(mask)

    final_response_str = [_last_assistant_text(chat) for chat in chat_states]
    full_chat_states = chat_states

    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_path = os.path.join(LOG_DIR, f"log.json")

    # one JSON-line per conversation so it's easy to stream-read
    timestamp = datetime.utcnow().isoformat()
    with open(log_path, "a") as f:
        for conv in full_chat_states:
            f.write(json.dumps({
                "utc_ts": timestamp,
                "conversation": conv,        # ⬅ everything - system / user / tool / assistant
            }) + "\n")

    agentic_outputs = AgenticOutputs(prompt_tokens=prompt_toks, response_tokens=response_toks, response_masks=response_masks, final_response_str=final_response_str, full_chat_states=full_chat_states)

    return agentic_outputs


def parse_flag_from_text(text):
    """Return the final complete flag in text, never only its prefix."""
    return extract_flag(text)


def compute_correctness_reward(gold_answer, generated_answer):
    gold_flag = parse_flag_from_text(gold_answer)
    generated_flag = parse_flag_from_text(generated_answer)
    if gold_flag is None or generated_flag is None:
        return 0.0
    return 1.0 if gold_flag == generated_flag else 0.0


def _last_assistant_text(chat):
    for msg in reversed(chat["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""   # no assistant/tool message found


def accuracy_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['answer']
    completion_contents  = [_last_assistant_text(c) for c in completions]
    rewards = []
    for sample, gold in zip(completion_contents, solutions):
        reward = compute_correctness_reward(gold_answer=gold, generated_answer=sample)
        rewards.append(reward)
    return rewards


def reward_tool_syntax(prompts, completions, **reward_kwargs):
    """
    +0.2  if *all* JSON objects produced by every assistant message:
        • are valid dicts
        • contain both "name"  and "inputs" keys
        • "name" is one of the registered mcp tools
    0   otherwise
    """
    tool_names = {t["name"] for t in mcp.list_tools_sync()}
    scores = []
    for comp in completions:
        ok = True
        final_has_box = False
        for msg in comp["messages"]:
            if msg["role"] != "assistant":
                continue
            if re.search(r"\\boxed\\{flag\\{[^}]+\\}\\}", msg["content"]):
                final_has_box = True
            for obj in extract_json_objects(msg["content"]):
                # basic structural checks
                if not (isinstance(obj, dict) and
                        "name" in obj and
                        "inputs" in obj and
                        obj["name"] in tool_names):
                    ok = False
                    break
            if not ok:
                break
        scores.append(0.2 if ok and not final_has_box else 0.0)
    return scores


def reward_answer_format(prompts, completions, **reward_kwargs):
    """
    +0.1  if the *last* assistant message contains    \\boxed{flag{…}}
          **and** that same message contains **no** JSON tool call.
    0   otherwise.
    """
    scores = []
    for comp in completions:
        final_text = _last_assistant_text(comp)
        has_flag  = bool(FLAG_BOX_RE.search(final_text))
        has_call  = bool(extract_json_objects(final_text))
        scores.append(0.1 if has_flag and not has_call else 0.0)
    return scores


def reward_python_execution(prompts, completions, reward_value=0.3, **reward_kwargs):
    """Reward at least one successful Python execution in a trajectory."""
    scores = []
    for comp in completions:
        tool_messages = [message for message in comp["messages"] if message["role"] == "tool"]
        if not tool_messages:
            scores.append(0.0)
            continue

        had_execution = False
        any_error = False
        for message in tool_messages:
            content = message.get("content", [])
            text = ""
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = str(content[0].get("text", ""))
            elif isinstance(content, str):
                text = content
            if text.startswith("Output:"):
                had_execution = True
            else:
                any_error = True
        scores.append(float(reward_value) if had_execution and not any_error else 0.0)
    return scores


def _canonical_messages(messages, model_id):
    canonical = []
    for turn, message in enumerate(messages):
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        canonical.append(
            {
                "turn": turn,
                "role": message.get("role", "unknown"),
                "content": content,
                "model": model_id if message.get("role") == "assistant" else None,
            }
        )
    return canonical


def run_eval_pass_majority(
    model,
    test_dataset,
    generate_fn,
    tokenizer,
    k: int = 8,
    max_generations: int = 4,
    max_new_tokens: int = 8192,
    *,
    output_dir,
    run_id,
    model_id,
    backend,
    split="eval",
    policy=None,
):
    """Run k attempts and persist exact-match metrics and JSONL episodes."""
    if k < 1:
        raise ValueError("k must be at least 1")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_path / "trajectories.jsonl"
    trajectories_path.write_text("", encoding="utf-8")

    questions = list(test_dataset["prompt"])
    gold_answers = list(test_dataset["answer"])
    task_ids = list(test_dataset["task_id"])
    difficulties = list(test_dataset["difficulty"])
    if not questions:
        raise ValueError("Evaluation dataset is empty")

    predictions = {task_id: [] for task_id in task_ids}
    gold_by_task = dict(zip(task_ids, gold_answers))
    policy = policy or f"solo:{model_id}"

    for sample_index in range(k):
        print(f"\n=== Generation {sample_index + 1}/{k} ===")
        agentic_output = model.run_agent(
            generate_fn,
            tokenizer,
            questions,
            max_generations=max_generations,
            max_new_tokens=max_new_tokens,
        )
        round_rewards = accuracy_reward(
            questions,
            agentic_output.full_chat_states,
            answer=gold_answers,
        )

        with trajectories_path.open("a", encoding="utf-8") as trajectory_file:
            for index, response in enumerate(agentic_output.final_response_str):
                task_id = task_ids[index]
                prediction = parse_flag_from_text(response)
                predictions[task_id].append(prediction)
                chat = agentic_output.full_chat_states[index]
                messages = _canonical_messages(chat["messages"], model_id)
                found_flags = []
                for message in messages:
                    found_flags.extend(extract_flags(message["content"]))
                episode = {
                    "task_id": task_id,
                    "episode_id": f"{run_id}:{task_id}:{sample_index}",
                    "run_id": run_id,
                    "attempt": sample_index,
                    "policy": policy,
                    "model_id": model_id,
                    "backend": backend,
                    "split": split,
                    "category": "crypto",
                    "difficulty": difficulties[index],
                    "messages": messages,
                    "solved": bool(round_rewards[index]),
                    "steps": sum(message["role"] == "assistant" for message in messages),
                    "flags_found": found_flags,
                    "sandbox_id": os.environ.get("ATTACKBOX_NAME", "attackbox"),
                    "tokens_in": None,
                    "tokens_out": None,
                }
                trajectory_file.write(json.dumps(episode, ensure_ascii=False) + "\n")

    metrics = compute_eval_metrics(gold_by_task, predictions)
    metrics.update(
        {
            "run_id": run_id,
            "model_id": model_id,
            "backend": backend,
            "split": split,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "trajectories_path": str(trajectories_path),
        }
    )
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nRESULTS:")
    print(f"empirical pass@1: {metrics['pass_at_1']:.4f}")
    print(f"pass@{k}:          {metrics['pass_at_k']:.4f}")
    print(f"majority@{k}:      {metrics['majority_at_k']:.4f}")
    print(f"Artifacts:         {output_path}")
    print("=" * 40)
    return metrics, predictions
