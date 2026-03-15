"""
Compile a ``QueryPlan`` into parameterised SQL.

The compiler never touches the network — it is a pure function that
returns ``(sql_template, params, warnings)``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from .models import Column, Operator, QueryPlan
from .utils import sanitise_identifier

logger = logging.getLogger(__name__)

# Mileage bucketing expressions (25 000 km buckets)
_MILEAGE_BUCKET_SELECT = "(floor(mileage::int / 25000) * 25000) AS mileage_bucket"
_MILEAGE_BUCKET_GROUP = "floor(mileage::int / 25000) * 25000"


def compile_sql(
    plan: QueryPlan,
    *,
    mileage_range: Optional[Tuple[int, int]] = None,
    year_range: Optional[Tuple[int, int]] = None,
    keyword_fallback: Optional[List[str]] = None,
) -> Tuple[str, List[Any], List[str]]:
    """
    Translate *plan* into a parameterised SQL string.

    Parameters
    ----------
    mileage_range, year_range
        If provided, the corresponding column filter is replaced by a
        ``BETWEEN %s AND %s`` clause (the original filter is skipped).
    keyword_fallback
        Extra ``condition ILIKE`` terms OR-ed together.

    Returns
    -------
    (sql, params, warnings)
    """
    warnings: List[str] = []
    params: List[Any] = []
    select_parts: List[str] = []

    group_cols = [c.value for c in plan.group_by] if plan.group_by else []
    has_groups = bool(group_cols)

    # ── SELECT / aggregations ────────────────────────────────
    if plan.aggregations:
        if group_cols:
            select_parts.extend(
                _mileage_aware_select(col, has_groups) for col in group_cols
            )
        for agg in plan.aggregations:
            alias = f" AS {agg.alias}" if agg.alias else ""
            select_parts.append(
                f"{agg.function.value}({agg.column.value}){alias}"
            )
        # Warn about non-grouped selects
        if plan.select:
            dropped = [
                c.value for c in plan.select if c.value not in group_cols
            ]
            if dropped:
                warnings.append(
                    f"Dropped non-aggregated columns not in GROUP BY: "
                    f"{dropped}. Add them to 'group_by' if needed."
                )
    else:
        if plan.select:
            select_parts.extend(c.value for c in plan.select)
        else:
            select_parts.append("*")

    sql = f"SELECT {', '.join(select_parts)} FROM auction_lots"

    # ── WHERE ────────────────────────────────────────────────
    where_clauses, where_params = _build_where(
        plan, mileage_range, year_range, keyword_fallback
    )
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
        params.extend(where_params)

    # ── GROUP BY ─────────────────────────────────────────────
    if group_cols:
        formatted = [
            _MILEAGE_BUCKET_GROUP if col == "mileage" else col
            for col in group_cols
        ]
        sql += " GROUP BY " + ", ".join(formatted)

    # ── ORDER BY ─────────────────────────────────────────────
    if plan.order_by:
        parts = []
        for o in plan.order_by:
            col = o.column.value if isinstance(o.column, Column) else o.column
            col = sanitise_identifier(col)          # prevent injection
            parts.append(f"{col} {o.direction.value}")
        sql += " ORDER BY " + ", ".join(parts)

    # ── LIMIT ────────────────────────────────────────────────
    sql += " LIMIT %s"
    params.append(plan.limit)

    return sql, params, warnings


# ── private helpers ──────────────────────────────────────────


def _mileage_aware_select(col: str, has_groups: bool) -> str:
    """Replace bare ``mileage`` with a bucket expression when grouping."""
    if col == "mileage" and has_groups:
        return _MILEAGE_BUCKET_SELECT
    return col


def _build_where(
    plan: QueryPlan,
    mileage_range: Optional[Tuple[int, int]],
    year_range: Optional[Tuple[int, int]],
    keyword_fallback: Optional[List[str]],
) -> Tuple[List[str], List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    consumed: set[Column] = set()

    # Range overrides
    if mileage_range is not None:
        clauses.append("mileage::int BETWEEN %s AND %s")
        params.extend(mileage_range)
        consumed.add(Column.mileage)

    if year_range is not None:
        clauses.append("year BETWEEN %s AND %s")
        params.extend(year_range)
        consumed.add(Column.year)

    # Regular filters (skip consumed)
    for f in plan.filters or []:
        if f.column in consumed:
            continue
        clauses.append(f"{f.column.value} {f.operator.value} %s")
        params.append(f.value)

    # Keyword fallback on condition
    if keyword_fallback:
        or_parts = " OR ".join(["condition ILIKE %s"] * len(keyword_fallback))
        clauses.append(f"({or_parts})")
        params.extend(f"%{kw}%" for kw in keyword_fallback)

    return clauses, params