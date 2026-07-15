# Assignment 3: Build an Agent from Scratch

This repository contains the code, traces, and evaluators for Assignment 3:

- **Part 2**: a coding ReAct agent built from scratch
- **Part 3**: an MCP stock-exchange agent connected to a live MCP server

## Repository Structure

- `part2/agent.py` — ReAct coding agent implementation
- `part2/evaluate_traces.py` — evaluator/scorer for Part 2 traces
- `part2/traces/` — saved run traces for Part 2
- `part2/README.md` — Part 2 details and design notes
- `part3/agent.py` — MCP stock-exchange agent implementation
- `part3/evaluate_traces.py` — evaluator/scorer for Part 3 traces
- `part3/traces/` — saved run traces for Part 3
- `part3/model_comparison_traces/` — traces comparing different models on Part 3

## Run Commands

```bash
uv run python part2/agent.py
uv run python part2/evaluate_traces.py
uv run python part3/agent.py
uv run python part3/evaluate_traces.py
```

## Notes

- The project uses `uv` for running the Python scripts.
- The `.env` file is not included because it contains the MCP API key.
- Part 3 requires a local `.env` file with a valid `X-API-Key` for the MCP exchange server.
- The final report is submitted separately; this repository is provided for code/traces review.
