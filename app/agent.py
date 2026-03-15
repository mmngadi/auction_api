"""
LLM agent orchestration — non-streaming and streaming paths.

Public API
──────────
    run_agent(prompt)          → str              (blocking, full answer)
    run_agent_stream(prompt)   → Generator[str]   (yields tokens for SSE)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Generator, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from .config import settings
from .tools import query_auction_data
from .utils import make_json_safe

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# THINK-TAG HANDLING (Qwen-3 series)
# ─────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")


def _extract_content(response) -> str:
    """Return visible text from a complete AIMessage, stripping <think> blocks."""
    raw: str = getattr(response, "content", None) or ""
    if not raw.strip():
        return ""
    outside = _THINK_RE.sub("", raw).strip()
    if outside:
        return outside
    match = re.search(r"<think>([\s\S]+)</think>", raw)
    if match:
        logger.warning("All content inside <think> tags — using as fallback.")
        return match.group(1).strip()
    return raw.strip()


class _ThinkFilter:
    """
    Streaming filter: buffers tokens until any ``<think>…</think>``
    block has passed, then forwards all subsequent tokens verbatim.
    """

    __slots__ = ("_buf", "_done")

    def __init__(self):
        self._buf: list[str] = []
        self._done = False

    def feed(self, token: str) -> str:
        """Feed one token; returns text to emit (may be empty while buffering)."""
        if self._done:
            return token

        self._buf.append(token)
        joined = "".join(self._buf)

        # Content clearly doesn't start with <think>
        stripped = joined.lstrip()
        if len(stripped) >= 7 and not stripped.startswith("<think>"):
            self._done = True
            self._buf.clear()
            return joined

        # End of think block found
        if "</think>" in joined:
            self._done = True
            self._buf.clear()
            return joined.split("</think>", 1)[1].lstrip()

        return ""

    def flush(self) -> str:
        """Return any remaining buffered text at end-of-stream."""
        if not self._buf:
            return ""
        joined = "".join(self._buf)
        self._buf.clear()
        if "<think>" in joined and "</think>" not in joined:
            return joined.split("<think>", 1)[1].strip()
        return joined


# ─────────────────────────────────────────────────────────────
# PROMPT CONSTANTS
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
# ─────────────────────────────────────────────────────────────

_llm = ChatOllama(
    model=settings.ollama.chat_model,
    base_url=settings.ollama.base_url,
    temperature=0,
    num_predict=settings.ollama.num_predict,
)
_llm_with_tools = _llm.bind_tools(_TOOLS)


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────


def run_agent(prompt: str, history: list[dict] | None = None) -> str:
    """Non-streaming: returns the complete answer string."""
    messages, tool_result, direct = _run_tool_loop(prompt, history)
    if tool_result is not None:
        return _summarise(prompt, tool_result, messages)
    return direct or "No response generated — please rephrase your question."


def run_agent_stream(
    prompt: str, history: list[dict] | None = None
) -> Generator[str, None, None]:
    """Streaming: yields text chunks as the summarisation LLM produces them."""
    messages, tool_result, direct = _run_tool_loop(prompt, history)
    if tool_result is not None:
        yield from _summarise_stream(prompt, tool_result, messages)
    else:
        yield direct or "No response generated — please rephrase your question."

# ─────────────────────────────────────────────────────────────
# TOOL LOOP  (shared by both paths)
# ─────────────────────────────────────────────────────────────


def _run_tool_loop(
    prompt: str,
    history: list[dict] | None = None,
) -> Tuple[list, Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute the agent's tool-calling loop.

    If conversation history is provided the LLM sees all prior turns,
    giving it context for follow-up questions like "What about 2020?".
    """
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    if history:
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            # "system" messages from the client are ignored —
            # we use our own SYSTEM_PROMPT
    else:
        # Backward compatibility: no history, just the prompt
        messages.append(HumanMessage(content=prompt))

    best_result: Optional[Dict[str, Any]] = None
    latest_result: Optional[Dict[str, Any]] = None
    tool_invoked = False

    for iteration in range(settings.max_agent_iterations):
        response = _llm_with_tools.invoke(messages)
        messages.append(response)

        has_tc = bool(getattr(response, "tool_calls", None))
        logger.debug(
            "Iter %d | tool_calls=%s | content_len=%d",
            iteration,
            has_tc,
            len(getattr(response, "content", "") or ""),
        )

        # ── No tool calls → exit loop ───────────────────────
        if not has_tc:
            effective = best_result or latest_result
            if tool_invoked and effective is not None:
                return messages, effective, None
            return messages, None, _extract_content(response)

        # ── Execute each tool call ───────────────────────────
        tool_invoked = True
        for tc in response.tool_calls:
            logger.info("Tool call → %s", tc.get("name"))
            result = _invoke_tool(tc)
            latest_result = make_json_safe(result)

            if isinstance(latest_result, dict) and "error" not in latest_result:
                best_result = latest_result

            if isinstance(latest_result, dict):
                logger.debug(
                    "Tool result | rows=%s | error=%s",
                    len(latest_result.get("rows", [])),
                    latest_result.get("error", "–"),
                )

            messages.append(
                ToolMessage(
                    content=json.dumps(latest_result),
                    tool_call_id=tc.get("id"),
                )
            )

    # Iteration limit reached
    effective = best_result or latest_result
    if effective is not None:
        return messages, effective, None
    return messages, None, "Agent reached its iteration limit."


# ─────────────────────────────────────────────────────────────
# SUMMARISATION  (non-streaming)
# ─────────────────────────────────────────────────────────────


def _summarise(
    prompt: str,
    tool_result: Dict[str, Any],
    messages: list,
) -> str:
    content = _build_summarisation_prompt(prompt, tool_result)
    messages.append(HumanMessage(content=content))

    final = _llm.invoke(messages)
    answer = _extract_content(final)

    if answer:
        return answer

    logger.warning("Summarisation returned empty content.")
    return _fallback_text(tool_result)


# ─────────────────────────────────────────────────────────────
# SUMMARISATION  (streaming)
# ─────────────────────────────────────────────────────────────


def _summarise_stream(
    prompt: str,
    tool_result: Dict[str, Any],
    messages: list,
) -> Generator[str, None, None]:
    """Yield text tokens from the summarisation LLM, stripping think-tags."""
    content = _build_summarisation_prompt(prompt, tool_result)
    messages.append(HumanMessage(content=content))

    think_filter = _ThinkFilter()
    yielded = False

    for chunk in _llm.stream(messages):
        text = chunk.content or ""
        if not text:
            continue
        filtered = think_filter.feed(text)
        if filtered:
            yield filtered
            yielded = True

    remaining = think_filter.flush()
    if remaining:
        yield remaining
        yielded = True

    if not yielded:
        yield _fallback_text(tool_result)


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────


def _fallback_text(tool_result: Dict[str, Any]) -> str:
    """Last-resort plain-text when the LLM produces nothing."""
    rows = tool_result.get("rows", [])
    if rows:
        preview = json.dumps(rows[:5], indent=2)
        return (
            f"Retrieved {len(rows)} record(s). "
            f"Sample:\n```json\n{preview}\n```"
        )
    if "error" in tool_result:
        return f"Query failed: {tool_result['error']}"
    return "No matching records found."


def _build_summarisation_prompt(
    prompt: str, tool_result: Dict[str, Any]
) -> str:
    rows = tool_result.get("rows", [])
    warnings = tool_result.get("warnings", [])
    broadened = any("broadened" in w.lower() for w in warnings)
    has_error = "error" in tool_result

    warnings_block = ""
    if warnings:
        lines = "\n".join(f"  • {w}" for w in warnings)
        warnings_block = f"\nSQL-compiler warnings:\n{lines}\n"

    if has_error:
        return (
            f"User question:\n{prompt}\n\n"
            f"The query encountered an error:\n  {tool_result['error']}\n\n"
            f"{warnings_block}"
            "Explain the problem and suggest how to adjust the question.\n"
        )

    if not rows:
        return (
            f"User question:\n{prompt}\n\n"
            "The database returned **no matching records**.\n\n"
            "IMPORTANT: Do NOT provide general market advice.\n"
            f"{warnings_block}\n"
            "Suggest broadening the search.\n"
        )

    extras = ""
    if broadened:
        extras = "• Results include broadened keyword fallback.\n"
    return (
        f"User question:\n{prompt}\n\n"
        "Auction results (mileage ±25 000 km · year ±5 · "
        "semantic condition search):\n"
        f"{json.dumps(tool_result, indent=2)}\n\n"
        f"{warnings_block}"
        "Notes:\n"
        "• Mileage ranges are ±25 000 km.\n"
        "• Condition grouping may use semantic embeddings.\n"
        f"{extras}\n"
        "Using ONLY the data above, provide:\n"
        "• A fair bid / price range in Rands.\n"
        "• Assumptions and reasoning.\n"
        "• Uncertainty caveats.\n"
    )


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
    """Deserialise any JSON-string values that the LLM should have sent as dicts."""
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