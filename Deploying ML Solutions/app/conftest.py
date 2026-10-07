# Lets `python -m pytest` import api, agent and predictor from this folder,
# and keeps tests offline even if AGENT_MODEL is set in your shell.
import os

if not os.getenv("EVAL_WITH_LLM"):
    os.environ.pop("AGENT_MODEL", None)
