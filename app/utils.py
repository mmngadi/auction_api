"""Serialisation helpers and lightweight text utilities."""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List


def make_json_safe(obj: Any) -> Any:
    """Recursively convert ``Decimal``, ``date``, and ``datetime`` for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Return lower-cased alpha-numeric tokens of at least *min_length*."""
    if not text:
        return []
    return [
        tok
        for tok in re.split(r"[^a-zA-Z0-9]+", text.lower())
        if len(tok) >= min_length
    ]


def strip_to_int(value: str | int | float) -> int:
    """Strip every non-digit character and return an ``int``."""
    return int(re.sub(r"\D", "", str(value)))


def sanitise_identifier(name: str) -> str:
    """Remove anything that isn't alphanumeric or underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "", name)