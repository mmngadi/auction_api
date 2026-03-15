"""
LangChain tool definitions.
"""

from __future__ import annotations

import json                          # ← add
import logging
from typing import Any, Dict, List, Optional, Tuple, Union   # ← add Union

from langchain_core.tools import tool

from .compiler import compile_sql
from .config import settings
from .database import db
from .models import Column, Filter, Operator, QueryPlan
from .utils import extract_keywords, make_json_safe, strip_to_int

logger = logging.getLogger(__name__)
_qcfg = settings.query

_DEFAULT_SELECT = [
    Column.make, Column.year, Column.mileage,
    Column.condition, Column.winning_bid_amount,
]
_DEFAULT_GROUP_BY = [
    Column.make, Column.year, Column.mileage, Column.condition,
]


@tool
def query_auction_data(plan: Union[dict, str]) -> dict:   # ← str accepted
    """
    Execute a schema-validated query against auction_lots.

    Automatically applies:
    • Broad keyword extraction for the 'make' filter
    • Mileage ±25 000 km tolerance
    • Year ±5 year tolerance
    • Default SELECT / GROUP BY when omitted
    • Keyword fallback on 'condition' when too few rows match
    """
    # ── 0. Deserialise if the LLM sent a JSON string ────────
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError as exc:
            return {"error": f"Malformed plan JSON: {exc}"}

    if not isinstance(plan, dict):
        return {"error": f"Expected dict, got {type(plan).__name__}"}

    # ── 1. Defensive unwrap ──────────────────────────────────
    if "plan" in plan and isinstance(plan["plan"], dict):
        logger.debug("Unwrapping double-nested plan key")
        plan = plan["plan"]

    # ── 2. Validate ──────────────────────────────────────────
    try:
        qp = QueryPlan(**plan)
    except Exception as exc:
        logger.warning("QueryPlan validation failed: %s | keys=%s", exc, list(plan.keys()))
        return {"error": f"Invalid QueryPlan: {exc}"}

    warnings: List[str] = []

    # ── 3. Enrich ────────────────────────────────────────────
    _broaden_make_filter(qp)
    mileage_range = _extract_mileage_range(qp)
    year_range = _extract_year_range(qp)
    _apply_defaults(qp)

    logger.info(
        "Enriched plan | mileage_range=%s | year_range=%s | filters=%d",
        mileage_range, year_range, len(qp.filters or []),
    )

    # ── 4. Compile ───────────────────────────────────────────
    try:
        sql, params, comp_warnings = compile_sql(
            qp,
            mileage_range=mileage_range,
            year_range=year_range,
        )
        warnings.extend(comp_warnings)
    except Exception as exc:
        return {"error": f"Compilation failed: {exc}", "warnings": warnings}

    # ── 5. Execute ───────────────────────────────────────────
    result = _safe_execute(sql, params)
    if "error" in result:
        result.setdefault("warnings", []).extend(warnings)
        return result

    # ── 6. Fallback ──────────────────────────────────────────
    rows = result.get("rows", [])
    if len(rows) < _qcfg.min_fallback_rows and qp.condition_text:
        fallback_kws = extract_keywords(qp.condition_text, _qcfg.min_keyword_length)
        if fallback_kws:
            warnings.append("Few condition matches — applying keyword fallback.")
            fb_sql, fb_params, fb_warns = compile_sql(
                qp,
                mileage_range=mileage_range,
                year_range=year_range,
                keyword_fallback=fallback_kws,
            )
            warnings.extend(fb_warns)
            fb_result = _safe_execute(fb_sql, fb_params)
            if fb_result.get("rows"):
                result = fb_result
                warnings.append("Returned results using broadened keyword fallback.")

    if warnings:
        result.setdefault("warnings", []).extend(warnings)
    return make_json_safe(result)


# ── enrichment helpers ───────────────────────────────────────

def _broaden_make_filter(qp: QueryPlan) -> None:
    for f in qp.filters or []:
        if f.column == Column.make and isinstance(f.value, str):
            keywords = extract_keywords(f.value)
            if keywords:
                f.operator = Operator.ilike
                f.value = "%" + " ".join(keywords) + "%"
                logger.debug("Make filter broadened → %s", f.value)
            break


def _extract_mileage_range(qp: QueryPlan) -> Optional[Tuple[int, int]]:
    for f in qp.filters or []:
        if f.column == Column.mileage:
            try:
                val = strip_to_int(str(f.value))
                return (max(0, val - _qcfg.mileage_tolerance_km),
                        val + _qcfg.mileage_tolerance_km)
            except (ValueError, TypeError):
                return None
    return None


def _extract_year_range(qp: QueryPlan) -> Optional[Tuple[int, int]]:
    for f in qp.filters or []:
        if f.column == Column.year:
            try:
                val = int(f.value)                    # works for both int and "2019"
                return (val - _qcfg.year_tolerance,
                        val + _qcfg.year_tolerance)
            except (ValueError, TypeError):
                pass
    return None


def _apply_defaults(qp: QueryPlan) -> None:
    if not qp.select:
        qp.select = list(_DEFAULT_SELECT)
    # Only add GROUP BY when aggregations are present (avoids PG error)
    if qp.aggregations and not qp.group_by:
        qp.group_by = list(_DEFAULT_GROUP_BY)

    # ── guard against data-destroying limits ─────────────────
    if qp.limit < _qcfg.min_limit:
        logger.info(
            "Limit raised from %d → %d (configured minimum)",
            qp.limit, _qcfg.min_limit,
        )
        qp.limit = _qcfg.min_limit


def _safe_execute(sql: str, params: list) -> dict:
    try:
        return db.execute(sql, params)
    except Exception as exc:
        logger.exception("Query execution failed")
        return {"error": str(exc), "sql": sql}