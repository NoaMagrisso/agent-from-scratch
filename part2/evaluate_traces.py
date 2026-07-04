"""
part2/evaluate_traces.py
Deterministic evaluator for Part 2 ReAct agent traces.

Reads part2/traces/task_*.json and checks structure + per-task correctness.
Does NOT call the LLM, run the agent, modify traces, or touch sandbox files.

Usage:
    uv run python part2/evaluate_traces.py              # all traces
    uv run python part2/evaluate_traces.py --trace part2/traces/task_2.json
    uv run python part2/evaluate_traces.py --json
"""

import argparse
import json
import sys
from pathlib import Path

TRACES_DIR = Path(__file__).parent / "traces"


# ── trace helpers ─────────────────────────────────────────────────────────────

def load_trace(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tool_calls_in(messages: list[dict]) -> list[dict]:
    """Flatten every tool call from all assistant messages into a list of {name, args}."""
    result = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}") or "{}")
            except json.JSONDecodeError:
                args = {}
            result.append({"name": fn.get("name", ""), "args": args})
    return result


def tool_observations(messages: list[dict], tool_name: str) -> list[dict]:
    """
    Return parsed observations for every call to `tool_name`.
    Matches each tool call to the next tool message that follows its assistant message.
    """
    obs = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []):
                if tc.get("function", {}).get("name") == tool_name:
                    for j in range(i + 1, len(messages)):
                        if messages[j].get("role") == "tool":
                            try:
                                obs.append(json.loads(messages[j]["content"]))
                            except Exception:
                                obs.append({"raw": messages[j].get("content", "")})
                            break
        i += 1
    return obs


def final_answer(messages: list[dict]) -> str:
    """Content of the last assistant message that has no tool_calls."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content", "")
    return ""


# ── structure check ───────────────────────────────────────────────────────────

def check_structure(messages: list[dict]) -> tuple[list[str], list[str]]:
    """Return (issues, warnings). issues → FAIL, warnings → WARN."""
    issues: list[str] = []
    warnings: list[str] = []

    if not messages or messages[0].get("role") != "system":
        issues.append("Missing system message at index 0")

    if len(messages) < 2 or messages[1].get("role") != "user":
        issues.append("Missing user message at index 1")

    if not messages or messages[-1].get("role") != "assistant":
        issues.append("Last message is not from assistant")
    elif not messages[-1].get("content", "").strip():
        warnings.append("Final assistant answer is empty")

    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if i + 1 >= len(messages) or messages[i + 1].get("role") != "tool":
                issues.append(
                    f"Assistant tool_calls at index {i} not followed by a tool message"
                )

    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            content = m.get("content", "")
            if content.strip() and content.strip()[0] in "{[":
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    warnings.append(f"Tool message at index {i} contains invalid JSON")

    return issues, warnings


# ── task-specific evaluators ──────────────────────────────────────────────────

def evaluate_task_1(messages: list[dict]) -> tuple[list[str], list[str], list[str]]:
    t_issues: list[str] = []
    t_warns: list[str] = []
    c_issues: list[str] = []

    calls = tool_calls_in(messages)
    names = [c["name"] for c in calls]

    if "list_files" not in names:
        t_issues.append("list_files not called")

    read_paths = [c["args"].get("path") for c in calls if c["name"] == "read_file"]
    if "hello.py" not in read_paths:
        t_issues.append("read_file('hello.py') not called")

    run_paths = [c["args"].get("path") for c in calls if c["name"] == "run_python"]
    if "hello.py" not in run_paths:
        t_issues.append("run_python('hello.py') not called")

    run_obs = tool_observations(messages, "run_python")
    if not any("hello world" in str(o.get("stdout", "")) for o in run_obs):
        c_issues.append("run_python stdout did not contain 'hello world'")

    ans = final_answer(messages)
    if "hello world" not in ans.lower():
        c_issues.append("Final answer does not mention 'hello world'")

    return t_issues, t_warns, c_issues


def evaluate_task_2(messages: list[dict]) -> tuple[list[str], list[str], list[str]]:
    t_issues: list[str] = []
    t_warns: list[str] = []
    c_issues: list[str] = []

    calls = tool_calls_in(messages)
    names = [c["name"] for c in calls]

    read_paths = [c["args"].get("path") for c in calls if c["name"] == "read_file"]
    if "buggy.py" not in read_paths:
        t_issues.append("read_file('buggy.py') not called")

    if "write_file" not in names:
        t_issues.append("write_file not called")

    if "run_python" not in names:
        t_issues.append("run_python not called")

    write_paths = [c["args"].get("path") for c in calls if c["name"] == "write_file"]
    if write_paths and "buggy.py" not in write_paths:
        t_warns.append(
            f"write_file wrote to {write_paths} instead of 'buggy.py' (in-place edit preferred)"
        )

    run_obs = tool_observations(messages, "run_python")
    if not any("5" in str(o.get("stdout", "")) for o in run_obs):
        c_issues.append("run_python stdout did not contain '5'")

    ans = final_answer(messages)
    ans_lower = ans.lower()
    bug_mentioned = any(kw in ans_lower for kw in ("subtraction", "addition", "a - b", "a + b", "bug"))
    if not bug_mentioned:
        c_issues.append("Final answer does not explain the a-b vs a+b bug")

    return t_issues, t_warns, c_issues


def evaluate_task_3(messages: list[dict]) -> tuple[list[str], list[str], list[str]]:
    t_issues: list[str] = []
    t_warns: list[str] = []
    c_issues: list[str] = []

    calls = tool_calls_in(messages)

    write_paths = [c["args"].get("path") for c in calls if c["name"] == "write_file"]
    if "reverse.py" not in write_paths:
        t_issues.append("write_file('reverse.py') not called")

    run_paths = [c["args"].get("path") for c in calls if c["name"] == "run_python"]
    if "reverse.py" not in run_paths:
        t_issues.append("run_python('reverse.py') not called")

    run_obs = tool_observations(messages, "run_python")
    if not any("olleh" in str(o.get("stdout", "")) for o in run_obs):
        c_issues.append("run_python stdout did not contain 'olleh'")

    ans = final_answer(messages)
    if "olleh" not in ans.lower():
        c_issues.append("Final answer does not mention 'olleh'")

    return t_issues, t_warns, c_issues


def evaluate_task_4(messages: list[dict]) -> tuple[list[str], list[str], list[str]]:
    t_issues: list[str] = []
    t_warns: list[str] = []
    c_issues: list[str] = []

    calls = tool_calls_in(messages)

    search_patterns = [
        c["args"].get("pattern", "").lower()
        for c in calls
        if c["name"] == "search_files"
    ]
    if not any("todo" in p for p in search_patterns):
        t_issues.append("search_files('TODO') not called")

    search_obs = tool_observations(messages, "search_files")
    notes_in_results = any(
        any(m.get("file") == "notes.py" for m in o.get("matches", []))
        for o in search_obs
        if isinstance(o, dict)
    )
    if notes_in_results:
        read_paths = [c["args"].get("path") for c in calls if c["name"] == "read_file"]
        if "notes.py" not in read_paths:
            t_issues.append("notes.py appeared in search results but read_file('notes.py') not called")

    ans = final_answer(messages)
    if "add input validation" not in ans and "# TODO: add input validation" not in ans:
        c_issues.append("Final answer missing TODO: add input validation")
    if "write unit tests" not in ans and "# TODO: write unit tests" not in ans:
        c_issues.append("Final answer missing TODO: write unit tests")

    return t_issues, t_warns, c_issues


TASK_EVALUATORS = [
    evaluate_task_1,
    evaluate_task_2,
    evaluate_task_3,
    evaluate_task_4,
]


# ── verdict helpers ───────────────────────────────────────────────────────────

def _status(issues: list[str], warns: list[str]) -> str:
    if issues:
        return "FAIL"
    if warns:
        return "WARN"
    return "PASS"


def _verdict(
    struct_issues: list[str],
    struct_warns: list[str],
    tool_issues: list[str],
    tool_warns: list[str],
    correct_issues: list[str],
) -> str:
    if struct_issues or tool_issues or correct_issues:
        return "FAIL"
    if struct_warns or tool_warns:
        return "WARN"
    return "PASS"


# ── evaluate one trace ────────────────────────────────────────────────────────

def evaluate_one(task_num: int, path: Path) -> dict:
    try:
        messages = load_trace(path)
    except Exception as e:
        return {
            "task": task_num,
            "path": str(path),
            "structure": "FAIL",
            "tool_sequence": "FAIL",
            "correctness": "FAIL",
            "verdict": "FAIL",
            "notes": [f"Could not load trace: {e}"],
        }

    struct_issues, struct_warns = check_structure(messages)

    if 1 <= task_num <= len(TASK_EVALUATORS):
        t_issues, t_warns, c_issues = TASK_EVALUATORS[task_num - 1](messages)
    else:
        t_issues, t_warns, c_issues = [], [f"No evaluator defined for task {task_num}"], []

    v = _verdict(struct_issues, struct_warns, t_issues, t_warns, c_issues)

    all_notes = struct_issues + struct_warns + t_issues + t_warns + c_issues
    if not all_notes:
        all_notes = ["OK"]

    return {
        "task": task_num,
        "path": str(path),
        "structure": _status(struct_issues, struct_warns),
        "tool_sequence": _status(t_issues, t_warns),
        "correctness": _status(c_issues, []),
        "verdict": v,
        "notes": all_notes,
    }


# ── output ────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    col = [6, 12, 15, 13, 8]
    header = (
        f"{'Task':<{col[0]}} {'Structure':<{col[1]}} {'Tool Sequence':<{col[2]}} "
        f"{'Correctness':<{col[3]}} {'Verdict':<{col[4]}} Notes"
    )
    print(header)
    print("-" * (sum(col) + len(col) + 6))
    for r in results:
        notes_str = "; ".join(r["notes"])
        print(
            f"{r['task']:<{col[0]}} {r['structure']:<{col[1]}} {r['tool_sequence']:<{col[2]}} "
            f"{r['correctness']:<{col[3]}} {r['verdict']:<{col[4]}} {notes_str}"
        )


def print_summary(results: list[dict]) -> None:
    total = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_warn = sum(1 for r in results if r["verdict"] == "WARN")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    need_review = [str(r["task"]) for r in results if r["verdict"] in ("WARN", "FAIL")]

    print(f"\nEvaluated {total} task(s): {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")
    if need_review:
        print(f"Tasks needing review: {', '.join(need_review)}")
    else:
        print("All tasks passed — no manual review needed.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Part 2 ReAct agent traces (no LLM calls)."
    )
    parser.add_argument("--trace", metavar="FILE", help="Evaluate a single trace file")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    if args.trace:
        p = Path(args.trace)
        stem = p.stem  # e.g. "task_2"
        try:
            task_num = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            task_num = 0
        results = [evaluate_one(task_num, p)]
    else:
        trace_files = sorted(TRACES_DIR.glob("task_*.json"))
        if not trace_files:
            print(f"No trace files found in {TRACES_DIR}", file=sys.stderr)
            sys.exit(1)
        results = []
        for tf in trace_files:
            try:
                task_num = int(tf.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            results.append(evaluate_one(task_num, tf))

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_table(results)
        print_summary(results)


if __name__ == "__main__":
    main()
