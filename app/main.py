"""
FastAPI application entry-point.

Lifecycle
---------
* **startup** — opens the database pool.
* **shutdown** — drains connections gracefully.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent
from .config import settings
from .database import db
from .models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse

# ── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    db.open()
    logger.info("Application started  (v%s)", settings.app_version)
    yield
    db.close()
    logger.info("Application shut down.")


# ── Application ──────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
)
def health():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        database="connected" if db.healthy else "disconnected",
        version=settings.app_version,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["chat"],
)
def chat(req: ChatRequest):
    """
    Send a natural-language prompt and receive a data-backed auction
    recommendation.
    """
    try:
        answer = run_agent(req.prompt)
        return ChatResponse(answer=answer)
    except Exception:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail="Internal agent error.")