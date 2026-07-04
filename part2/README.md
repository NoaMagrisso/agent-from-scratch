# Part 2 — ReAct Coding Agent from Scratch

A minimal ReAct-style coding agent implemented in plain Python.  
No LangChain. No frameworks. Just the Ollama API, a tool dispatch loop, and five Python functions.

## What it implements

- **`dispatch_tool(name, args)`** — looks up a tool by name, calls it with `**args`, returns a structured error for unknown tools or exceptions.
- **`run_agent(goal)`** — the ReAct loop: sends messages + tool schemas to the LLM, appends every reply, dispatches tool calls, repeats until the model returns a final answer with no tool calls or `MAX_STEPS` is reached.
- **Trace saving** — each task's full message history is written to `part2/traces/task_<N>.json` as a flat list of OpenAI-format chat messages.

## Model / Provider

| Setting | Value |
|---|---|
| Provider | Ollama (OpenAI-compatible API) |
| Base URL | `http://localhost:11434/v1` |
| Model | `granite4:micro` |

Pull the model before running: `ollama pull granite4:micro`

## Available Tools

| Tool | Signature | Description |
|---|---|---|
| `list_files` | `() → list[str]` | Lists all files in the in-memory sandbox |
| `read_file` | `(path) → {path, content}` | Reads a sandbox file |
| `write_file` | `(path, content) → {path, written}` | Creates or overwrites a sandbox file |
| `run_python` | `(path) → {path, stdout}` | Runs a sandbox file via `exec()` and captures stdout |
| `search_files` | `(pattern) → {matches}` | Case-insensitive substring search across all sandbox files |

## How the ReAct Loop Works

```
messages = [system_prompt, user_goal]

for step in 1 .. MAX_STEPS:
    resp = LLM(messages, tools=TOOLS)
    messages.append(resp.message)

    if resp has NO tool_calls:
        return resp.content          ← final answer

    for each tool_call:
        args = parse_json(tool_call.arguments)
        result = dispatch_tool(name, args)
        messages.append({role: "tool", content: json(result)})

return "(stopped: hit MAX_STEPS)"
```

## Running the Agent

```bash
# Ensure Ollama is running, then:
uv run python part2/agent.py
```

Runs all four tasks sequentially and saves traces to `part2/traces/`.

## Running the Evaluator

```bash
# Evaluate all traces (no LLM calls)
uv run python part2/evaluate_traces.py

# Evaluate a single trace
uv run python part2/evaluate_traces.py --trace part2/traces/task_2.json

# Machine-readable JSON output
uv run python part2/evaluate_traces.py --json
```

## Traces

Saved to `part2/traces/task_<N>.json` after each run. Each file is a flat list of OpenAI-format messages (`system`, `user`, `assistant`, `tool`).

## The Four Tasks

| # | Goal | What it tests |
|---|---|---|
| 1 | List files → read `hello.py` → run it | Tool chaining: list → read → run |
| 2 | Read `buggy.py` → fix bug in place → run | Code editing and debugging |
| 3 | Write `reverse.py` → run it | Code generation |
| 4 | Search for `TODO` → read matching files | Search + conditional file read |

## Evaluator Output Columns

| Column | Meaning |
|---|---|
| Structure | System/user messages present; assistant-tool pairing correct |
| Tool Sequence | Required tools called in the right order with correct arguments |
| Correctness | Tool outputs and final answer contain expected values |
| Verdict | `PASS` / `WARN` / `FAIL` — `WARN` means non-critical deviation |
