"""
Part 3 — Wire Your Loop to a LIVE MCP Exchange
===============================================
Same loop as Part 2 — and the same two kinds of tools, side by side:

    Part 2:  tools = TOOL_FN dict          (Python functions in your file)
    Part 3:  tools = ONE MCP server  +  a local TOOL_FN helper

The MCP server is the real competition exchange at agent-stocks.vercel.app.
Prices are live US-market prices, the leaderboard is shared with every other
team, and orders move real (pretend) money. Nothing is deterministic anymore —
so your agent must read what actually happened and never fake a result.

Alongside the exchange your agent also gets one *local* Python tool —
`pct_change`, a pure compute helper (no network) that reports the percent move
between two prices. It's the same dict-of-functions you wrote in Part 2; it
just lives next to the MCP tools in one flat list the model sees. You may ADD
more local tools (function + TOOL_FN + LOCAL_TOOLS entry).

Three roles, same as always:
    Ollama (granite4:micro) = the BRAIN   (model provider, text in / text out)
    MCP exchange + local    = the HANDS   (tools the agent can call)
    THIS FILE               = the LOOP    (run_agent — the code YOU write)

You implement the same building blocks you would for any MCP agent — the same
shape as Part 2, plus the MCP twist:
    TODO 1. mcp_tools_to_openai(mcp_tools)  — translate the server's tool list
            into the OpenAI-compatible chat-completions tool schema (same shape
            you wrote by hand in Part 2's TOOLS list). Ollama, OpenAI, vLLM,
            LiteLLM, etc. all accept this same shape.
    TODO 2. dispatch_tool(name, args, ...)  — like Part 2's dispatch_tool, but a
            tool is now EITHER a local Python function (TOOL_FN) OR a remote MCP
            tool (await client.call_tool). Route each to the right place.
    TODO 3. run_agent(goal, client)         — the observe -> think -> act loop.
            It merges BOTH kinds of tools into one list and calls dispatch_tool
            on each requested tool call.

Everything else — the trading system prompt, the local helper, the goals, the
goal-driving driver `run_goals`, the trace writer, .env loading — is provided.
The driver runs each instructor-set GOAL once; for every goal the *model*
decides what to do, and you submit the trace it produced.

Before you write code, do Part 3A: explore the exchange by hand with the MCP
Inspector

Setup (Part 3B):
    1. Register your team ON THE WEBSITE (https://agent-stocks.vercel.app) to
       get your API key. Do NOT register from code.
    2. Put the key in part3/.env (already in .gitignore):
           AGENTS_EXCHANGE_API_KEY=ax_your_key_here
    3. Run:
           uv run python part3/agent.py            # run every goal
           uv run python part3/agent.py --only 3   # re-run just goal 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

# Force UTF-8 output so debug prints don't crash on Windows with non-ASCII chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Ollama speaks the OpenAI chat-completions protocol on localhost:11434/v1.
_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = "granite4:micro"  # pinned: verified to support native tool calling
MAX_STEPS = 12            # hard stop so a confused model can't loop forever
DEBUG = True              # set False to silence per-step prints

OUR_TEAM = "score100-or-die"  # registered team name — never change, never guess

# The real competition exchange. All tools require your X-API-Key header.
LIVE_URL = "https://agent-stocks.vercel.app/api/mcp"

# The live exchange rate-limits trading calls (12 s between them). We pace each
# goal well clear of that — never retry in a tight loop.
SECONDS_BETWEEN_CYCLES = 15

TRACES_DIR = Path(__file__).parent / "traces"

# The trading policy lives in the system prompt — the MODEL makes the buy/sell/
# hold call. Tune these rules to shape how your agent behaves.
SYSTEM_PROMPT = f"""\
You are an autonomous trading agent for team "{OUR_TEAM}" on a live stock exchange.
You are given one goal at a time. Use the provided tools, then reply with a plain-text
summary. Place AT MOST ONE order per goal.

MONEY RULES — all *_cents fields are integers in cents, NOT dollars:
- NEVER put a "$" sign in front of a raw cents value (e.g. never write "$36680").
- ALL tool observations that contain *_cents values also contain pre-computed dollar strings
  in fields like "last_dollars", "net_worth_dollars", "_cash_dollars", "price_dollars".
  USE THOSE FIELDS — do not do any arithmetic yourself.

PRE-COMPUTED FIELDS IN TOOL RESPONSES (use these directly):

  get_portfolio response includes:
    "_cash_dollars" — cash balance as a dollar string, e.g. "$90,314.92"
    For positions: call get_quote(symbol), then position_value(qty, price_cents=last_cents).

  get_symbols response includes:
    "_extremes.most_expensive" — symbol, name, price_dollars of the priciest stock
    "_extremes.cheapest"       — symbol, name, price_dollars of the cheapest stock
    Use these directly. Do NOT call find_extreme_symbol or find_symbol_extremes.

  get_news response includes:
    "_headlines.formatted" — exact headline text and symbol for each item, one per line.
    Copy this field VERBATIM into your final answer. Do NOT reword any headline.

  get_quote response includes:
    "_summary.last_dollars"       — current price as a dollar string
    "_summary.prev_close_dollars" — previous close as a dollar string
    Use these in your answer, not the raw last_cents.

  get_leaderboard response includes:
    "_our_team.rank"              — our 1-based rank (integer)
    "_our_team.net_worth_dollars" — our net worth as a dollar string
    "_our_team.total_teams"       — total number of teams
    Report ONLY these fields for our team. Do not read other rows or count rows yourself.

GOAL-SPECIFIC SEQUENCES:

  PORTFOLIO REPORT:
    1. get_portfolio → use _cash_dollars for cash
    2. For EACH position: get_quote(symbol) → use _summary; then position_value(qty, last_cents)
    3. Report only tool-derived values. Never guess position values.

  PRICE COMPARISON:
    1. get_quote(AAPL) → use _summary.last_dollars
    2. get_quote(MSFT) → use _summary.last_dollars
    3. pct_change(old_cents=AAPL.last_cents, new_cents=MSFT.last_cents)

  BUY ORDER:
    1. get_portfolio → check _cash_dollars
    2. get_news → read _headlines; pick the symbol with tradeable=true and most positive news
       (skip any symbol where tradeable=false, e.g. "MKT" is market news — not a stock)
    3. get_quote(symbol) → use _summary.last_dollars for the price
       If get_quote returns an error, the symbol is invalid — pick the next tradeable symbol from _headlines
    4. estimate_buy_cost(price_cents=last_cents from get_quote, qty)
    5. place_order(side="buy", symbol, qty)
    6. trade_fill_summary(order_result from place_order)
    7. get_portfolio → confirm updated _cash_dollars
    Report ONLY from trade_fill_summary and the post-trade portfolio.

SAFETY — you MUST follow these at all times:
- Do NOT call place_order unless the goal explicitly asks you to trade.
- Never overdraw: verify total cost with estimate_buy_cost before buying.
- Only sell shares you actually hold (no short selling).
- Quantities must be whole shares.
- If place_order returns {{"error": ...}}, read it and adapt — never claim success.
- Our team is "{OUR_TEAM}". For leaderboard: use _our_team fields only.

When the goal is satisfied, reply with a short summary and make no further tool call.\
"""


# Tracing — so you can SEE what the agent does, AND save the trajectory.
# trace() both prints (watch the run live) and records into _TRACE, so the
# provided save_trace() can write each goal's trace as an agentevals-format
# message list — exactly like Part 2. You don't touch this; just keep calling
# trace("action", ...) / trace("observation", ...) from your loop.
_TRACE: list[dict] = []


def trace(step: str, payload) -> None:
    """Emit one structured trace line and record it for save_trace()."""
    print(f"  [{step}] {json.dumps(payload, default=str)[:300]}")
    _TRACE.append({"step": step, "payload": payload})


# ---------------------------------------------------------------------------
# Local tool(s) — the "regular tools" that live alongside the MCP exchange.
# A local tool is a plain Python function (no network), exactly like Part 2.
# run_agent merges these into the same flat tool list as the MCP tools, then
# dispatches them with a direct Python call (no call_tool, no await).
#
# Just one is provided: pct_change, a price-move helper. Add more of your own by
# writing the function, registering it in TOOL_FN, and describing it in
# LOCAL_TOOLS — the same two-step you did in Part 2.
# ---------------------------------------------------------------------------
def pct_change(old_cents: int, new_cents: int) -> dict:
    """
    Percent change from old_cents to new_cents (e.g. 100 -> 110 is +10.0).
    Read-only math the model can use to judge a price move.
    """
    if not old_cents:
        return {"error": "old_cents must be non-zero"}
    return {"pct": round((new_cents - old_cents) / old_cents * 100, 2)}


def cents_to_dollars(cents: int) -> dict:
    """Convert a cents integer to a human-readable dollar string."""
    dollars = cents / 100
    return {"dollars_str": f"${dollars:,.2f}", "dollars": dollars}


def find_extreme_symbol(items: list, mode: str) -> dict:
    """
    From the 'items' list returned by get_symbols, find the symbol with the
    highest (mode='most_expensive') or lowest (mode='cheapest') last_cents.
    Returns the full row plus a human-readable price_dollars field.
    """
    if not items:
        return {"error": "items list is empty"}
    if mode not in ("cheapest", "most_expensive"):
        return {"error": "mode must be 'cheapest' or 'most_expensive'"}
    chosen = (min if mode == "cheapest" else max)(
        items, key=lambda x: x.get("last_cents", 0)
    )
    return {
        "symbol": chosen.get("symbol"),
        "name": chosen.get("name"),
        "last_cents": chosen.get("last_cents"),
        "price_dollars": cents_to_dollars(chosen.get("last_cents", 0))["dollars_str"],
        "mode": mode,
    }


def find_team_on_leaderboard(items: list) -> dict:
    """
    Search the 'items' list from get_leaderboard for our team (score100-or-die).
    Returns rank (1-based), net_worth_dollars, and total number of teams.
    Returns an error dict if our team is not present.
    """
    for rank, entry in enumerate(items, 1):
        if entry.get("team_name") == OUR_TEAM:
            nw = entry.get("net_worth_cents", 0)
            return {
                "rank": rank,
                "total_teams": len(items),
                "team_name": OUR_TEAM,
                "net_worth_cents": nw,
                "net_worth_dollars": cents_to_dollars(nw)["dollars_str"],
            }
    return {
        "error": f"Team '{OUR_TEAM}' not found in leaderboard",
        "teams_present": [e.get("team_name") for e in items],
    }


def estimate_buy_cost(price_cents: int, qty: int) -> dict:
    """
    Calculate the total cost of buying qty shares at price_cents, including the
    0.05% trading fee. Use this BEFORE place_order to verify affordability.
    """
    if price_cents <= 0 or qty <= 0:
        return {"error": "price_cents and qty must be positive integers"}
    subtotal = price_cents * qty
    fee = round(subtotal * 0.0005)
    total = subtotal + fee
    return {
        "price_cents": price_cents,
        "qty": qty,
        "subtotal_cents": subtotal,
        "fee_cents": fee,
        "total_cents": total,
        "subtotal_dollars": cents_to_dollars(subtotal)["dollars_str"],
        "fee_dollars": cents_to_dollars(fee)["dollars_str"],
        "total_dollars": cents_to_dollars(total)["dollars_str"],
    }


def position_value(qty: int, price_cents: int) -> dict:
    """
    Calculate the current market value of a position: qty shares at price_cents.
    Use this to report what a holding is worth in dollars.
    """
    if qty <= 0 or price_cents <= 0:
        return {"error": "qty and price_cents must be positive integers"}
    total = qty * price_cents
    return {
        "qty": qty,
        "price_cents": price_cents,
        "price_dollars": cents_to_dollars(price_cents)["dollars_str"],
        "total_cents": total,
        "total_dollars": cents_to_dollars(total)["dollars_str"],
    }


def find_symbol_extremes(items: list, **_) -> dict:
    """
    From get_symbols 'items', return BOTH the most expensive and cheapest symbol
    in a single call with dollar-formatted prices. Use for any goal that asks for
    both extremes — never make two find_extreme_symbol calls.
    """
    if not items:
        return {"error": "items list is empty"}
    most_exp = max(items, key=lambda x: x.get("last_cents", 0))
    cheapest = min(items, key=lambda x: x.get("last_cents", 0))
    return {
        "most_expensive": {
            "symbol": most_exp.get("symbol"),
            "name": most_exp.get("name"),
            "last_cents": most_exp.get("last_cents"),
            "price_dollars": cents_to_dollars(most_exp.get("last_cents", 0))["dollars_str"],
        },
        "cheapest": {
            "symbol": cheapest.get("symbol"),
            "name": cheapest.get("name"),
            "last_cents": cheapest.get("last_cents"),
            "price_dollars": cents_to_dollars(cheapest.get("last_cents", 0))["dollars_str"],
        },
    }


def format_news_headlines(items: list, limit: int = 3) -> dict:
    """
    Return the top `limit` news headlines exactly as the exchange returned them.
    The 'formatted' field is a ready-to-paste string — copy it verbatim into
    your final answer. Do NOT paraphrase any headline.
    Symbols listed as 'MKT' are market-wide news, not individual stocks — do
    not attempt to buy or quote 'MKT'.
    """
    chosen = items[:limit]
    pairs = [
        {
            "headline": item.get("headline", ""),
            "symbol": item.get("symbol", ""),
            "tradeable": item.get("symbol", "") not in ("MKT",),
        }
        for item in chosen
    ]
    formatted = "\n".join(f'{p["headline"]} – {p["symbol"]}' for p in pairs)
    return {"headlines": pairs, "formatted": formatted}


def quote_summary(symbol: str, last_cents: int, prev_close_cents: int) -> dict:
    """
    Convert a raw get_quote result into human-readable dollar strings.
    Call this before displaying any price — never put $ in front of raw cents.
    """
    return {
        "symbol": symbol,
        "last_dollars": cents_to_dollars(last_cents)["dollars_str"],
        "prev_close_dollars": cents_to_dollars(prev_close_cents)["dollars_str"],
        "last_cents": last_cents,
        "prev_close_cents": prev_close_cents,
    }


def trade_fill_summary(order_result: dict) -> dict:
    """
    Format a place_order result into readable fill data with dollar prices.
    Pass the entire place_order response. Use this — not estimate_buy_cost — to
    report the actual fill price and total cost in the final answer.
    """
    if "error" in order_result:
        return {"error": order_result["error"]}
    fills = order_result.get("fills", [])
    if not fills:
        return {"error": "No fills returned", "raw": order_result}
    total_qty = sum(f.get("qty", 0) for f in fills)
    total_cost_cents = sum(f.get("qty", 0) * f.get("price_cents", 0) for f in fills)
    avg_price_cents = round(total_cost_cents / total_qty) if total_qty else 0
    return {
        "order_id": order_result.get("order_id"),
        "status": order_result.get("status"),
        "qty_filled": order_result.get("qty_filled", total_qty),
        "avg_price_dollars": cents_to_dollars(avg_price_cents)["dollars_str"],
        "total_cost_dollars": cents_to_dollars(total_cost_cents)["dollars_str"],
        "fills": [
            {
                "qty": f.get("qty"),
                "price_dollars": cents_to_dollars(f.get("price_cents", 0))["dollars_str"],
            }
            for f in fills
        ],
    }


# Local tool registry — name -> Python function. The "billboard" the model
# reads is LOCAL_TOOLS (the JSON schema); dispatch happens through TOOL_FN.
TOOL_FN: dict = {
    "pct_change": pct_change,
    "cents_to_dollars": cents_to_dollars,
    "find_extreme_symbol": find_extreme_symbol,
    "find_symbol_extremes": find_symbol_extremes,
    "find_team_on_leaderboard": find_team_on_leaderboard,
    "estimate_buy_cost": estimate_buy_cost,
    "position_value": position_value,
    "format_news_headlines": format_news_headlines,
    "quote_summary": quote_summary,
    "trade_fill_summary": trade_fill_summary,
}

LOCAL_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "pct_change",
            "description": (
                "Percent change between two cents prices. "
                "Use to compare MSFT vs AAPL current prices: "
                "old_cents=AAPL.last_cents, new_cents=MSFT.last_cents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_cents": {"type": "integer", "description": "baseline price in cents (e.g. AAPL last_cents)"},
                    "new_cents": {"type": "integer", "description": "comparison price in cents (e.g. MSFT last_cents)"},
                },
                "required": ["old_cents", "new_cents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cents_to_dollars",
            "description": (
                "Convert any *_cents integer to a dollar string. "
                "ALWAYS call this — never compute dollars yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cents": {"type": "integer", "description": "Value in cents, e.g. 10000000 → '$100,000.00'"},
                },
                "required": ["cents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_extreme_symbol",
            "description": (
                "Find the cheapest or most expensive symbol from get_symbols 'items'. "
                "Pass the full items array; set mode to 'cheapest' or 'most_expensive'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The 'items' array from get_symbols result",
                        "items": {"type": "object"},
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["cheapest", "most_expensive"],
                        "description": "Which extreme to find",
                    },
                },
                "required": ["items", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_team_on_leaderboard",
            "description": (
                f"Find our team '{OUR_TEAM}' in the get_leaderboard 'items'. "
                "Returns rank, net_worth_dollars, and total teams. "
                "ALWAYS use this for leaderboard tasks — never guess the row yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The 'items' array from get_leaderboard result",
                        "items": {"type": "object"},
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_buy_cost",
            "description": (
                "Calculate total cost (with 0.05% fee) for a buy order. "
                "Call this BEFORE place_order to verify you can afford it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price_cents": {"type": "integer", "description": "Per-share price in cents from get_quote"},
                    "qty": {"type": "integer", "description": "Number of whole shares to buy"},
                },
                "required": ["price_cents", "qty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "position_value",
            "description": (
                "Calculate what a position is worth: qty × price_cents. "
                "Use this to report holding values — never multiply in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qty": {"type": "integer", "description": "Number of shares held"},
                    "price_cents": {"type": "integer", "description": "Current price per share in cents from get_quote"},
                },
                "required": ["qty", "price_cents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol_extremes",
            "description": (
                "For market-survey goals asking for BOTH most expensive AND cheapest: "
                "call this ONCE with the full 'items' array from get_symbols. "
                "Returns both extremes with dollar-formatted prices. "
                "Do NOT use find_extreme_symbol or scan the list yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The 'items' array from get_symbols",
                        "items": {"type": "object"},
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_news_headlines",
            "description": (
                "For news-summary goals: pass the 'items' from get_news. "
                "Returns the top headlines EXACTLY as returned by the exchange. "
                "Copy the 'formatted' field verbatim into your final answer — do NOT paraphrase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The 'items' array from get_news",
                        "items": {"type": "object"},
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many headlines to return (default 3)",
                        "default": 3,
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quote_summary",
            "description": (
                "Convert a get_quote result into dollar-formatted strings. "
                "Call this BEFORE displaying any price. "
                "Never put $ in front of raw cents (e.g. $36680 is wrong — call this first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "last_cents": {"type": "integer", "description": "last_cents from get_quote"},
                    "prev_close_cents": {"type": "integer", "description": "prev_close_cents from get_quote"},
                },
                "required": ["symbol", "last_cents", "prev_close_cents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trade_fill_summary",
            "description": (
                "Format a place_order result into human-readable fill data. "
                "Call this AFTER place_order and use its output in the final answer. "
                "Do NOT report cost from estimate_buy_cost if the actual fill differs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_result": {
                        "type": "object",
                        "description": "The full response object from place_order",
                    },
                },
                "required": ["order_result"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# TODO 1 — mcp_tools_to_openai
# ---------------------------------------------------------------------------
def mcp_tools_to_openai(mcp_tools) -> list[dict]:
    """
    Translate the tool definitions returned by client.list_tools() into the
    OpenAI-compatible chat-completions tool schema (what
    _client.chat.completions.create(tools=...) expects). Ollama, OpenAI, vLLM,
    LiteLLM, etc. all accept this same shape.

    Each MCP tool object has:
        .name        (str)
        .description (str or None)
        .inputSchema (dict — already a valid JSON Schema, or None)

    Each tool entry must look like:
        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": ...,   # the inputSchema, or {"type":"object","properties":{}}
            }
        }

    Hint: this is the same shape you wrote by hand in Part 2's TOOLS list.
    """
    result = []
    for tool in mcp_tools:
        # Support both attribute-style objects and plain dicts
        if hasattr(tool, "name"):
            name = tool.name
            description = getattr(tool, "description", "") or ""
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
        else:
            name = tool.get("name", "")
            description = tool.get("description", "") or ""
            schema = (
                tool.get("inputSchema")
                or tool.get("input_schema")
                or {"type": "object", "properties": {}}
            )
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        })
    return result


# ---------------------------------------------------------------------------
# TODO 2 — dispatch_tool  (Part 2's dispatch, extended for MCP)
# ---------------------------------------------------------------------------
async def dispatch_tool(name: str, args: dict, client: Client, mcp_names: set):
    """
    Route ONE tool call to where the tool actually lives, and return its result
    (or {"error": ...} on failure). This is Part 2's dispatch_tool, extended:
    a tool is now EITHER a local Python function OR a remote MCP tool.

      - if name in TOOL_FN:   a plain Python call,  TOOL_FN[name](**args)
      - elif name in mcp_names: result = await client.call_tool(name, dict(args))
                                then return result.data  (NOT a plain value —
                                MCP wraps the payload in .data)
      - else:                 an unknown tool — return an {"error": ...} dict

    `mcp_names` is the set of names the MCP server advertised, built in run_agent
    from client.list_tools(), so you can tell a local tool from an MCP tool.
    Returning {"error": ...} (instead of raising) keeps a bad call as a normal
    observation the model can read and adapt to — just like Part 2.
    """
    try:
        if name in TOOL_FN:
            if DEBUG:
                print(f"  [dispatch] local Python tool: {name!r}")
            return TOOL_FN[name](**args)
        elif name in mcp_names:
            if DEBUG:
                print(f"  [dispatch] remote MCP tool: {name!r}")
            # raise_on_error=False so errors come back as observations, not exceptions.
            # This keeps the error JSON the exchange returns (e.g. {"error": "..."})
            # intact rather than wrapping it in a ToolError string.
            result = await client.call_tool(name, args, raise_on_error=False)

            # fastmcp CallToolResult: prefer .data (structured), then .content (text)
            if result.data is not None:
                return result.data

            # Extract text from the content block list
            content = result.content or []
            texts = []
            for item in content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, dict):
                    texts.append(item.get("text", str(item)))
                else:
                    texts.append(str(item))
            combined = "\n".join(texts)
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, TypeError):
                return combined if combined else {"error": f"Tool '{name}' returned an error"}
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        if DEBUG:
            print(f"  [dispatch] tool {name!r} raised: {e}")
        return {"error": str(e), "tool": name}


# ---------------------------------------------------------------------------
# TODO 3 — run_agent  (the loop — identical to Part 2 except dispatch)
# ---------------------------------------------------------------------------
def _auto_augment(tool_name: str, result: object) -> object:
    """
    After certain MCP calls, enrich the result with pre-computed helper output
    so the model has formatted data right in the observation — it does not need
    a separate helper call, and cannot misread raw cents as dollars.
    """
    if not isinstance(result, dict):
        return result
    result = dict(result)  # shallow copy so we don't mutate the original
    if tool_name == "get_news" and "items" in result:
        result["_headlines"] = format_news_headlines(result["items"])
    elif tool_name == "get_leaderboard" and "items" in result:
        result["_our_team"] = find_team_on_leaderboard(result["items"])
    elif tool_name == "get_portfolio" and "cash_cents" in result:
        result["_cash_dollars"] = cents_to_dollars(result["cash_cents"])["dollars_str"]
    elif tool_name == "get_symbols" and "items" in result:
        result["_extremes"] = find_symbol_extremes(result["items"])
    elif tool_name == "get_quote" and "last_cents" in result:
        sym = result.get("symbol", "")
        result["_summary"] = quote_summary(sym, result["last_cents"], result.get("prev_close_cents", 0))
    return result


async def run_agent(goal: str, client: Client) -> tuple[str, list[dict]]:
    """
    Drive one goal to completion. Returns (final_answer, tool_log).

    client: a single ALREADY-OPEN MCP client (the exchange). The tools the model
    sees are the LOCAL_TOOLS plus the MCP server's tools, merged into one flat
    list. dispatch_tool routes each call to its local function or to the MCP
    client. main()/run_goals() owns the session lifecycle — run_agent never opens
    or closes the client.

    Steps:
      1. Get the MCP tools with `await client.list_tools()`. Build:
           - `tools`: the flat OpenAI-format list. Combine the local schemas
             with the translated MCP ones:  LOCAL_TOOLS + mcp_tools_to_openai(...)
           - `mcp_names`: a set of the MCP tool names, so dispatch_tool can tell
             local tools from MCP ones. Pass it (and `client`) to dispatch_tool.
      2. Run the same observe -> think -> act loop as Part 2. For each tool call:
           observation = await dispatch_tool(name, args, client, mcp_names)
      3. Record every executed call into `tool_log` as
           {"tool": name, "args": args, "result": observation}
         and return it alongside the final answer, so the driver can write a
         faithful trace of the goal.

    The rest — messages, the chat completion call, appending observations as
    tool-role messages with their tool_call_id — is identical to Part 2.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_log: list[dict] = []

    # 1. Discover MCP tools and build the merged tool list
    mcp_tool_list = await client.list_tools()
    mcp_names: set[str] = {
        t.name if hasattr(t, "name") else t["name"] for t in mcp_tool_list
    }
    if DEBUG:
        print(f"[setup] discovered MCP tools: {', '.join(sorted(mcp_names))}")
        print(f"[setup] local tools: {', '.join(sorted(TOOL_FN.keys()))}")

    all_tools = LOCAL_TOOLS + mcp_tools_to_openai(mcp_tool_list)

    # 2. ReAct loop — identical structure to Part 2
    for step in range(1, MAX_STEPS + 1):
        if DEBUG:
            print(f"\n[step {step}] calling model")

        resp = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=all_tools,
        )
        msg = resp.choices[0].message

        # Build a serialisable plain dict from the SDK response object
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if DEBUG and msg.content:
            print(f"[step {step}] assistant: {msg.content[:300]}")

        # 3. No tool calls → model is done
        if not msg.tool_calls:
            if DEBUG:
                print(f"[step {step}] no tool calls — returning final answer")
            return msg.content or "", tool_log

        # 4. Dispatch each requested tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or ""
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(f"[step {step}] JSON parse error for {name!r}: {e}")
                error_result = {"error": f"JSON parse error in arguments: {e}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(error_result, ensure_ascii=False),
                })
                continue

            if DEBUG:
                print(f"[step {step}] tool call: {name}({json.dumps(args)[:200]})")
            trace("action", {"tool": name, "args": args})

            result = await dispatch_tool(name, args, client, mcp_names)
            # Enrich MCP results with pre-formatted helper data
            if name in mcp_names:
                result = _auto_augment(name, result)

            if DEBUG:
                print(f"[step {step}] observation: {json.dumps(result, default=str)[:300]}")
            trace("observation", result)

            tool_log.append({"tool": name, "args": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    if DEBUG:
        print(f"[reached MAX_STEPS={MAX_STEPS}] stopping")
    return f"(stopped: hit MAX_STEPS={MAX_STEPS} without a final answer)", tool_log


# ---------------------------------------------------------------------------
# The goals — the instructor sets these; the agent runs each one once and you
# submit the trace it produces. Add your own goals at the end to exercise more
# of the agent; the driver runs every goal in this list.
# (Provided — you do not need to change this, though you may add goals.)
# ---------------------------------------------------------------------------
GOALS = [
    # 1. Read-only: report the portfolio.
    "Report your current portfolio: your cash in dollars and every position"
    " you hold.",

    # 2. Read-only: survey the market.
    "Survey the market: list the available symbols with their current prices,"
    " and tell me which one is the most expensive and which is the cheapest.",

    # 3. Read-only + local tool: compare two symbols by price.
    "Compare AAPL and MSFT: quote both and use pct_change to say how far"
    " MSFT's price is above or below AAPL's. Do not trade.",

    # 4. Read-only: summarize the latest news.
    "Read the latest news and summarize, in one line each, the three most"
    " recent headlines and which symbol each is about.",

    # 5. Trade: a news-driven buy, sized conservatively.
    "Buy the stock with the most supportive recent news. Spend at most 30% of"
    " your net worth on it and keep at least 20% of your portfolio in cash."
    " Place the order and confirm the fill from the result.",

    # 6. Trade: prune holdings that no longer have supporting news.
    "Review your holdings and sell any position you can no longer justify from"
    " recent news. If every holding is still justified, hold and explain why.",

    # 7. Read-only: where do we stand against everyone else.
    "Check the leaderboard and tell me our team's rank and net worth relative"
    " to the other teams. Do not trade.",
]


# ---------------------------------------------------------------------------
# Trace logging — the DRIVER owns this, not the model. One file per goal.
# These files are your evidence; the grader cross-checks them against the
# exchange's get_trades history, so they must reflect what really happened.
# Same agentevals format as Part 2: a flat list of OpenAI-format chat messages.
# (Provided — you do not need to change this.)
# ---------------------------------------------------------------------------
def save_trace(goal_num: int, goal: str, answer: str) -> Path:
    """
    Write the trajectory recorded in _TRACE for one goal to
    part3/traces/goal_<N>.json, as a flat list of OpenAI-format chat messages.

    This is the shape LangChain's `agentevals` expects for trajectory match
    evaluators (https://github.com/langchain-ai/agentevals) — identical to
    Part 2's task_<N>.json: each tool call is an assistant message with a
    `tool_calls` entry whose `arguments` is a JSON string, followed by a
    `tool`-role message holding the result. Returns the path.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    for entry in _TRACE:
        if entry["step"] == "action":
            call = entry["payload"]  # {"tool": name, "args": {...}}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": call.get("tool"),
                                "arguments": json.dumps(call.get("args", {}), default=str),
                            }
                        }
                    ],
                }
            )
        elif entry["step"] == "observation":
            messages.append(
                {"role": "tool", "content": json.dumps(entry["payload"], default=str)}
            )
    # the model's final natural-language answer (no tool call)
    messages.append({"role": "assistant", "content": answer})

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / f"goal_{goal_num}.json"
    path.write_text(json.dumps(messages, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# run_goals — the outer driver. Deterministic Python around the LLM loop, the
# Part 3 twin of Part 2's run_task loop: for each goal, clear the trace, run the
# agent, save the trace. The only Part-3 additions are the shared MCP session
# (opened once, since there is no sandbox to reset) and the rate-limit sleep the
# live exchange requires. The saved traces are what you submit.
# (Provided — you do not need to change this, though you may add goals.)
# ---------------------------------------------------------------------------
async def run_goals(api_key: str, goals: list[tuple[int, str]]) -> None:
    """
    Run each goal in `goals` once against the live exchange. `goals` is a list
    of (goal_num, goal_text) pairs — goal_num is the 1-based position in GOALS,
    so the saved filename matches the goal even with --only N. For each goal:
      1. clear _TRACE (fresh trajectory — the MCP analog of resetting Part 2's
         sandbox; the live exchange has no sandbox to reset);
      2. run_agent() lets the model inspect the market and act on the goal;
      3. write part3/traces/goal_<N>.json;
      4. sleep SECONDS_BETWEEN_CYCLES to stay clear of the rate limit.
    """
    async with AsyncExitStack() as stack:
        exchange = await stack.enter_async_context(make_live_client(api_key))

        for idx, (num, goal) in enumerate(goals):
            print(f"\n{'='*60}\nGOAL {num}: {goal}\n{'='*60}")
            _TRACE.clear()  # fresh trajectory per goal (no sandbox to reset)
            try:
                answer, _tool_log = await run_agent(goal, exchange)
            except NotImplementedError:
                raise
            except Exception as e:  # never let one bad goal kill the run
                print(f"  [goal-error] {e!r}")
                answer = f"(goal errored: {e})"

            path = save_trace(num, goal, answer)
            print(f"\n--- ANSWER: {answer}")
            print(f"--- trace saved to {path}")

            if idx < len(goals) - 1:
                time.sleep(SECONDS_BETWEEN_CYCLES)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def make_live_client(api_key: str) -> Client:
    """Connect to the real competition exchange with your API key header."""
    return Client(
        StreamableHttpTransport(url=LIVE_URL, headers={"X-API-Key": api_key})
    )


def load_api_key() -> str:
    """Read AGENTS_EXCHANGE_API_KEY from the environment or part3/.env."""
    api_key = os.environ.get("AGENTS_EXCHANGE_API_KEY", "")
    if not api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("AGENTS_EXCHANGE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    return api_key


async def main():
    parser = argparse.ArgumentParser(description="Live MCP trading agent")
    parser.add_argument("--only", type=int, default=None, metavar="N",
                        help="run only goal N (1-based) instead of all goals")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: set AGENTS_EXCHANGE_API_KEY in your environment or part3/.env")
        print("       (register your team on https://agent-stocks.vercel.app to get a key)")
        return

    # (goal_num, goal_text) pairs — goal_num is the 1-based position in GOALS so
    # the trace filename matches the goal even when running a single --only goal.
    goals = list(enumerate(GOALS, 1))
    if args.only is not None:
        if not 1 <= args.only <= len(GOALS):
            print(f"ERROR: --only must be between 1 and {len(GOALS)}")
            return
        goals = [(args.only, GOALS[args.only - 1])]

    await run_goals(api_key, goals)


if __name__ == "__main__":
    asyncio.run(main())
