"""
LLM agent orchestration.

Fixes vs. v1
─────────────
• Strips Qwen-3 <think> blocks so visible content is never empty.
• Uses a **plain** LLM (no tools) for the final summarisation pass.
• Adds DEBUG-level breadcrumbs for every iteration, tool result and
  content preview — visible with LOG_LEVEL=DEBUG.
• Graceful fallback when the model still produces nothing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from .config import settings
from .tools import query_auction_data
from .utils import make_json_safe

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# THINK-TAG HANDLING  (Qwen-3 series)
# ─────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")


def _extract_content(response) -> str:
    """
    Return the *visible* answer from an ``AIMessage``.

    Qwen-3 models emit ``<think>…</think>`` blocks before the answer.
    If the model accidentally puts **all** text inside the tags (and
    nothing outside), we fall back to the thinking content itself.
    """
    raw: str = getattr(response, "content", None) or ""
    if not raw.strip():
        return ""

    # 1. Strip think blocks → keep everything outside
    outside = _THINK_RE.sub("", raw).strip()
    if outside:
        return outside

    # 2. Fallback: use the content *inside* the think block
    match = re.search(r"<think>([\s\S]+)</think>", raw)
    if match:
        logger.warning("All content was inside <think> tags — using it as fallback.")
        return match.group(1).strip()

    return raw.strip()


# ─────────────────────────────────────────────────────────────
# PROMPT CONSTANTS  (unchanged from v1 — repeated for completeness)
# ─────────────────────────────────────────────────────────────

DATABASE_SCHEMA = """\
TABLE: auction_lots

Columns:
  id                     INTEGER PRIMARY KEY
  status_of_lot          TEXT
  winning_bid_amount     INTEGER
  date_of_auction        DATE
  year                   INTEGER
  make                   TEXT
  category               TEXT
  trade_amount           TEXT
  mileage                TEXT
  service_book           TEXT
  no_of_keys             INTEGER
  colour                 TEXT
  condition              TEXT
  condition_embedding    VECTOR(768)
"""

_QUERY_PLAN_EXAMPLE = {
    "plan": {
        "select": ["make", "year"],
        "aggregations": [
            {
                "function": "AVG",
                "column": "winning_bid_amount",
                "alias": "average_price",
            }
        ],
        "filters": [
            {"column": "make", "operator": "ILIKE", "value": "%Toyota%"}
        ],
        "group_by": ["make"],
        "order_by": [{"column": "average_price", "direction": "DESC"}],
        "limit": 20,
        "condition_text": "optional natural language condition search",
    }
}

SYSTEM_PROMPT = f"""\
You are an AI assistant for vehicle auctions.

Rules
─────
• Always retrieve real data via `query_auction_data`.
• Never write raw SQL — produce a QueryPlan dict instead.
• Prefer aggregations (AVG, MIN, MAX, COUNT) over post-processing.
• Use `condition_text` for semantic vehicle-condition search.
• Follow the schema exactly:

{DATABASE_SCHEMA}

Behaviour
─────────
1. If the user misspells a make / model, correct it before querying
   and acknowledge the correction.
2. NEVER fabricate estimates when no data is returned.  State clearly
   that the database has no matching records and suggest broadening
   the search.
3. If the tool reports dropped non-aggregated columns, fix your plan
   by adding them to `group_by` or removing them from `select`.

Example QueryPlan:
{json.dumps(_QUERY_PLAN_EXAMPLE, indent=2)}
"""

# ─────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────

_TOOL_MAP: Dict[str, Any] = {"query_auction_data": query_auction_data}
_TOOLS = list(_TOOL_MAP.values())

# ─────────────────────────────────────────────────────────────
# LLM INSTANCES
#
#   _llm            → plain model for free-text summarisation
#   _llm_with_tools → same model with tools bound (agent loop)
# ─────────────────────────────────────────────────────────────

_llm = ChatOllama(
    model=settings.ollama.chat_model,
    base_url=settings.ollama.base_url,
    temperature=0,
    num_predict=4096,
)
_llm_with_tools = _llm.bind_tools(_TOOLS)

# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────


def run_agent(prompt: str) -> str:
    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    best_result: Optional[Dict[str, Any]] = None   # ← last *successful* result
    latest_result: Optional[Dict[str, Any]] = None  # ← most recent (may be error)
    tool_invoked = False

    for iteration in range(settings.max_agent_iterations):
        response = _llm_with_tools.invoke(messages)
        messages.append(response)

        has_tool_calls = bool(getattr(response, "tool_calls", None))
        logger.debug(
            "Iter %d | tool_calls=%s | content_len=%d | preview=%.300s",
            iteration,
            has_tool_calls,
            len(getattr(response, "content", "") or ""),
            (getattr(response, "content", "") or "")[:300],
        )

        if not has_tool_calls:
            # Prefer the successful result; fall back to latest
            effective = best_result or latest_result
            if tool_invoked and effective is not None:
                return _summarise(prompt, effective, messages)
            content = _extract_content(response)
            return content or "No response generated — please rephrase your question."

        tool_invoked = True
        for tc in response.tool_calls:
            logger.info(
                "Tool call → %s | arg_keys=%s",
                tc.get("name"),
                list(tc.get("args", {}).keys()),
            )
            result = _invoke_tool(tc)
            latest_result = make_json_safe(result)

            # Track best (non-error) result
            if isinstance(latest_result, dict) and "error" not in latest_result:
                best_result = latest_result

            if isinstance(latest_result, dict):
                logger.debug(
                    "Tool result | rows=%s | error=%s | warnings=%s",
                    len(latest_result.get("rows", [])),
                    latest_result.get("error", "-"),
                    latest_result.get("warnings", []),
                )

            messages.append(
                ToolMessage(
                    content=json.dumps(latest_result),
                    tool_call_id=tc.get("id"),
                )
            )

    return "The agent reached its iteration limit without a final answer."


# ─────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────


def _invoke_tool(tool_call: dict) -> Any:
    fn = _TOOL_MAP.get(tool_call.get("name", ""))
    if fn is None:
        return {"error": f"Unknown tool: {tool_call.get('name')}"}

    args = _coerce_tool_args(tool_call.get("args", {}))

    try:
        return fn.invoke(args)
    except Exception as exc:
        logger.exception("Tool invocation failed")
        return {"error": str(exc)}


def _coerce_tool_args(args: dict) -> dict:
    """
    LLMs occasionally emit a JSON *string* where a dict is expected.
    Detect this and ``json.loads`` it so Pydantic validation passes.
    """
    out = {}
    for key, value in args.items():
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    value = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        out[key] = value
    return out


def _summarise(
    prompt: str,
    tool_result: Dict[str, Any],
    messages: list,
) -> str:
    """Build a summarisation prompt, invoke the *plain* LLM, and return text."""
    content = _build_summarisation_prompt(prompt, tool_result)
    messages.append(HumanMessage(content=content))

    final = _llm.invoke(messages)  # ← no tools bound
    answer = _extract_content(final)

    if answer:
        return answer

    # ── Last-resort fallback ─────────────────────────────────
    logger.warning("Summarisation produced empty content — returning raw data.")
    rows = tool_result.get("rows", [])
    if rows:
        preview = json.dumps(rows[:5], indent=2)
        return (
            f"I retrieved {len(rows)} auction record(s) but could not "
            f"generate a narrative summary.  Here is a sample:\n\n```json\n{preview}\n```"
        )
    if "error" in tool_result:
        return f"The database query failed: {tool_result['error']}"
    return "The database returned no matching records for your query."


def _build_summarisation_prompt(prompt: str, tool_result: Dict[str, Any]) -> str:
    """Return the human-message text that asks the LLM to summarise."""
    rows = tool_result.get("rows", [])
    warnings = tool_result.get("warnings", [])
    broadened = any("broadened" in w.lower() for w in warnings)
    has_error = "error" in tool_result

    warnings_block = ""
    if warnings:
        lines = "\n".join(f"  • {w}" for w in warnings)
        warnings_block = f"\nSQL-compiler warnings:\n{lines}\n"

    # ── query error ──────────────────────────────────────────
    if has_error:
        return (
            f"User question:\n{prompt}\n\n"
            f"The query encountered an error:\n  {tool_result['error']}\n\n"
            f"{warnings_block}"
            "Explain the problem in plain language and suggest how the "
            "user can adjust their question (spelling, broader ranges, etc.).\n"
        )

    # ── no rows ──────────────────────────────────────────────
    if not rows:
        return (
            f"User question:\n{prompt}\n\n"
            "The database returned **no matching records**.\n\n"
            "IMPORTANT: Do NOT provide general market advice — only "
            "data-backed recommendations are permitted.\n"
            f"{warnings_block}\n"
            "Suggest the user:\n"
            "• Confirm the make / model spelling.\n"
            "• Widen the year range.\n"
            "• Broaden the mileage tolerance.\n"
            "• Search using condition keywords.\n"
        )

    # ── has rows → full summarisation ────────────────────────
    extras = ""
    if broadened:
        extras = (
            "• Some results came from a broadened keyword fallback — "
            "reliability is lower.\n"
        )
    return (
        f"User question:\n{prompt}\n\n"
        "Auction results (mileage ±25 000 km · year ±5 · semantic "
        "condition search):\n"
        f"{json.dumps(tool_result, indent=2)}\n\n"
        f"{warnings_block}"
        "Interpretation notes:\n"
        "• Mileage ranges are ±25 000 km.\n"
        "• Condition grouping may use semantic embeddings.\n"
        f"{extras}\n"
        "Using ONLY the data above, provide:\n"
        "• A fair bid / price range **in Rands**.\n"
        "• Assumptions and reasoning.\n"
        "• Uncertainty caveats and data-sparseness notes.\n"
    )