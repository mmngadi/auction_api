"""
OpenAI-compatible Chat Completions API.

Endpoints
─────────
    POST /v1/chat/completions   — streaming + non-streaming chat
    GET  /v1/models             — list available models
    GET  /health                — healthcheck
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Generator, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent import run_agent, run_agent_stream
from .config import settings
from .database import db

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = settings.model_id
    messages: List[Message]
    stream: bool = False
    # Accepted for SDK compatibility — not used by the agent
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = None
    user: Optional[str] = None


class _Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[_Choice]
    usage: _Usage = Field(default_factory=_Usage)


# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.open()
    logger.info("Application started (v%s)", settings.app_version)
    yield
    db.close()
    logger.info("Application stopped")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────


async def _verify_api_key(request: Request):
    """Validate ``Authorization: Bearer <key>`` when keys are configured."""
    if not settings.api_keys:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_oai_error("Missing Bearer token.", "invalid_api_key"),
        )
    if auth.removeprefix("Bearer ") not in settings.api_keys:
        raise HTTPException(
            status_code=401,
            detail=_oai_error("Invalid API key.", "invalid_api_key"),
        )


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS  (OpenAI error envelope)
# ─────────────────────────────────────────────────────────────


def _oai_error(message: str, code: str | None = None, type_: str = "api_error"):
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    body = (
        exc.detail
        if isinstance(exc.detail, dict)
        else _oai_error(str(exc.detail))
    )
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(Exception)
async def _generic_exc(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=_oai_error("Internal server error", type_="server_error"),
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.model_id,
        "version": settings.app_version,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "auction-api",
            }
        ],
    }


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(_verify_api_key)],
)
async def chat_completions(req: ChatCompletionRequest):
    prompt = _extract_prompt(req.messages)
    history = _build_history(req.messages)

    # ── streaming ────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, req.model, history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── non-streaming ────────────────────────────────────────
    answer = run_agent(prompt, history)
    return ChatCompletionResponse(
        id=_chat_id(),
        created=_now(),
        model=req.model,
        choices=[_Choice(message=Message(role="assistant", content=answer))],
    )

# ── Helpers ──────────────────────────────────────────────────

def _extract_prompt(messages: List[Message]) -> str:
    """Return the last user message content."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(
        status_code=400,
        detail=_oai_error(
            "No user message found in messages array.",
            code="missing_user_message",
            type_="invalid_request_error",
        ),
    )


def _build_history(messages: List[Message]) -> list[dict]:
    """Convert the full conversation to simple dicts for the agent."""
    return [{"role": m.role, "content": m.content} for m in messages]

def _chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


# ── SSE helpers ──────────────────────────────────────────────


def _sse(payload: Any) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(
    chat_id: str,
    created: int,
    model: str,
    *,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def _stream_response(
    prompt: str, model: str, history: list[dict]
) -> Generator[str, None, None]:
    """Sync generator — FastAPI runs this in a threadpool automatically."""
    cid = _chat_id()
    ts = _now()

    yield _sse(_chunk(cid, ts, model, delta={"role": "assistant"}))

    try:
        for token in run_agent_stream(prompt, history):
            if token:
                yield _sse(_chunk(cid, ts, model, delta={"content": token}))
    except Exception:
        logger.exception("Error during streaming")
        yield _sse(
            _chunk(cid, ts, model, delta={"content": "\n\n[Error generating response]"})
        )

    yield _sse(_chunk(cid, ts, model, delta={}, finish_reason="stop"))
    yield "data: [DONE]\n\n"