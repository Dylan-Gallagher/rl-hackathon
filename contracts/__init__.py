"""contracts — shared task/transcript/env schemas for the CTF Alloy Hackathon.

Everything here mirrors README §3 field-for-field. Do not fork these schemas;
propose changes in the group channel.
"""

from contracts.env.base import CTFEnv, Obs, capped_output
from contracts.env.daytona import DaytonaCTFEnv
from contracts.env.docker import DockerCTFEnv
from contracts.env.mock import MockCTFEnv
from contracts.env.repl import ReplCTFEnv
from contracts.flag import new_flag, scan_for_flags, verify_flag
from contracts.task import (
    Horizon,
    Task,
    TaskEnv,
    TaskFlag,
    load_tasks,
)
from contracts.transcript import (
    Transcript,
    TranscriptMessage,
    episode_id_for,
    iter_transcripts,
    write_transcript,
)

__all__ = [
    "CTFEnv",
    "Obs",
    "capped_output",
    "DaytonaCTFEnv",
    "DockerCTFEnv",
    "MockCTFEnv",
    "ReplCTFEnv",
    "Task",
    "TaskEnv",
    "TaskFlag",
    "Horizon",
    "load_tasks",
    "Transcript",
    "TranscriptMessage",
    "write_transcript",
    "iter_transcripts",
    "episode_id_for",
    "new_flag",
    "verify_flag",
    "scan_for_flags",
]
