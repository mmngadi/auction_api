"""Custom exception hierarchy — every layer raises a typed error."""


class AuctionError(Exception):
    """Base application error."""


class DatabaseError(AuctionError):
    """Database pool or execution failure."""


class SQLValidationError(AuctionError):
    """SQL safety-check failure."""


class QueryCompilationError(AuctionError):
    """QueryPlan → SQL compilation failure."""


class QueryPlanError(AuctionError):
    """Malformed QueryPlan."""


class EmbeddingError(AuctionError):
    """Embedding generation failure."""