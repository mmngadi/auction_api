"""Pydantic models, column / operator enumerations, and API schemas."""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ── Column catalogue (mirrors DB schema) ─────────────────────

class Column(str, Enum):
    id = "id"
    status_of_lot = "status_of_lot"
    winning_bid_amount = "winning_bid_amount"
    date_of_auction = "date_of_auction"
    year = "year"
    make = "make"
    category = "category"
    trade_amount = "trade_amount"
    mileage = "mileage"
    service_book = "service_book"
    no_of_keys = "no_of_keys"
    colour = "colour"
    condition = "condition"


# ── SQL building-blocks ──────────────────────────────────────

class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    gte = ">="
    lte = "<="
    like = "LIKE"
    ilike = "ILIKE"


class AggFunction(str, Enum):
    avg = "AVG"
    sum = "SUM"
    min = "MIN"
    max = "MAX"
    count = "COUNT"


class SortDirection(str, Enum):
    asc = "ASC"
    desc = "DESC"


class Aggregation(BaseModel):
    function: AggFunction
    column: Column
    alias: Optional[str] = None


class Filter(BaseModel):
    column: Column
    operator: Operator
    value: Union[str, int, float]


class OrderBy(BaseModel):
    column: Union[Column, str]
    direction: SortDirection = SortDirection.asc


# ── Semantic query plan ──────────────────────────────────────

class QueryPlan(BaseModel):
    """
    Declarative query plan produced by the LLM.

    Supports SELECT, AGGREGATE, FILTER, GROUP BY, ORDER BY, LIMIT
    and a free-text ``condition_text`` for semantic condition search.
    """

    select: Optional[List[Column]] = None
    aggregations: Optional[List[Aggregation]] = None
    filters: Optional[List[Filter]] = None
    group_by: Optional[List[Column]] = None
    order_by: Optional[List[OrderBy]] = None
    limit: Optional[int] = None
    condition_text: Optional[str] = None


# ── API request / response schemas ───────────────────────────

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str