"""path4.verl_grpo — GRPO-on-the-learnable-band training side (README Path 4 step 2).

Binary flag reward, DAPO-style dynamic sampling filter, EI pass-rate-band
curriculum, and a veRL-shaped async tool-executor adapter over `CTFEnv`.

Pipeline position: alloy → traces → SFT (`path4.coldstart`) → EI (Path 3)
→ **GRPO (this package)** → ensemble → scoreboard.
"""
