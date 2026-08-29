"""Alloy Agents — replication of XBOW's 'Agents Built From Alloys'.

Core idea: run a single agent loop over ONE shared conversation, but randomly
alternate which LLM generates each step. Each model sees prior assistant/tool
turns as its own, unaware a different model produced them. Total model-call
count is identical to a single-model run, yet solve rate rises when the two
models are diverse (cross-provider).
"""
