"""
Ollama embedding client.

Currently available for future semantic-condition ordering via
``condition_embedding <-> vector``.  Called explicitly when needed.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import ollama

from .config import settings
from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_client: Optional[ollama.Client] = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(
            host=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
        )
    return _client


def create_embedding(text: str) -> List[float]:
    """
    Generate a 768-dim vector for *text* using ``nomic-embed-text``.

    Raises
    ------
    EmbeddingError
        On any client / model failure.
    """
    try:
        resp = _get_client().embeddings(
            model=settings.ollama.embed_model,
            prompt=text,
        )
        embedding = resp.get("embedding")
        if embedding is None:
            raise EmbeddingError("Model returned no embedding.")
        return embedding
    except EmbeddingError:
        raise
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise EmbeddingError(str(exc)) from exc