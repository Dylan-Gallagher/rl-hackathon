"""Small open-weight model CTF baseline — SEPARATE from the alloy replication.

Serves Qwen3-8B (or comparable) with vLLM in a Daytona GPU sandbox and runs it
through the IDENTICAL agent harness (alloy.agent), tools, and turn cap used for
the frontier models, so no serving-stack difference (chat template, sampling,
tokenizer) confounds later fine-tuned-variant comparisons. Results live under
baseline/results/, kept apart from the alloy run.
"""
