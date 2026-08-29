"""path4.scoreboard — live Pass@k / Maj@k scoreboard over canonical transcripts.

It's just transcripts + a viewer (README §Path 4 step 4). Aggregation, FastAPI
server, static UI, CLI, and an offline demo seeder.
"""

from path4.scoreboard.metrics import aggregate, scan_race_summaries, scan_transcripts

__all__ = ["aggregate", "scan_transcripts", "scan_race_summaries"]
