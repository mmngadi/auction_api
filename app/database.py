"""Thread-safe PostgreSQL connection pool with lifecycle control."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

from psycopg2.pool import ThreadedConnectionPool

from .config import settings
from .exceptions import DatabaseError, SQLValidationError
from .utils import make_json_safe

logger = logging.getLogger(__name__)

_FORBIDDEN_KEYWORDS = frozenset(
    ["drop", "delete", "update", "insert", "alter", "truncate", "create"]
)


class DatabaseManager:
    """Wraps a ``ThreadedConnectionPool`` with validation and result mapping."""

    def __init__(self) -> None:
        self._pool: Optional[ThreadedConnectionPool] = None

    # ── lifecycle ────────────────────────────────────────────

    def open(self) -> None:
        """Create the connection pool (idempotent)."""
        if self._pool is not None:
            return
        logger.info("Opening DB pool → %s", settings.db.dsn)
        self._pool = ThreadedConnectionPool(
            settings.db.pool_min,
            settings.db.pool_max,
            settings.db.dsn,
        )

    def close(self) -> None:
        """Drain and release every connection."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("DB pool closed.")

    @property
    def healthy(self) -> bool:
        """Return ``True`` if a trivial query succeeds."""
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ── connection context manager ───────────────────────────

    @contextmanager
    def _connection(self):
        if self._pool is None:
            raise DatabaseError("Database pool is not initialised.")
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    # ── SQL validation ───────────────────────────────────────

    @staticmethod
    def _validate_sql(sql: str) -> None:
        lowered = sql.lower().strip()
        if not lowered.startswith("select"):
            raise SQLValidationError("Only SELECT queries are permitted.")
        for kw in _FORBIDDEN_KEYWORDS:
            if kw in lowered:
                raise SQLValidationError(f"Forbidden keyword detected: '{kw}'")

    # ── query execution ──────────────────────────────────────

    def execute(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate, execute, and return ``{sql, columns, rows}``
        where each row is a ``dict``.

        Raises
        ------
        SQLValidationError
            If the query contains forbidden DDL/DML.
        DatabaseError
            If the result set exceeds ``settings.query.max_rows``.
        """
        self._validate_sql(sql)
        logger.info("SQL  %s | params=%s", sql, params)

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]

        if len(rows) > settings.query.max_rows:
            raise DatabaseError(
                f"Result set too large ({len(rows)} rows; "
                f"limit is {settings.query.max_rows})."
            )

        return make_json_safe(
            {
                "sql": sql,
                "columns": columns,
                "rows": [dict(zip(columns, row)) for row in rows],
            }
        )


# Singleton — opened/closed via FastAPI lifespan
db = DatabaseManager()