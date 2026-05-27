"""FastAPI entrypoint for the standalone Graphiti RAG backend.

The app exposes only the minimum HTTP surface required by the Chainlit UI and
automation clients: health, text ingestion, chat, history retrieval, and
session reset. All repository-specific concerns are intentionally excluded.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.application import GraphitiApplication
from app.config import Settings
from app.models import (
    ChatRequest,
    ChatResponse,
    GroupsResponse,
    HealthResponse,
    HistoryResponse,
    IngestRequest,
    IngestResponse,
    ResetSessionRequest,
)


settings: Settings = Settings.from_env()
graphiti_app: GraphitiApplication = GraphitiApplication(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize storage-backed services once when the ASGI app starts."""

    await graphiti_app.startup()
    yield
    await graphiti_app.shutdown()


app: FastAPI = FastAPI(
    title="Graphiti Backend",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a minimal readiness payload for API consumers."""

    return graphiti_app.health()


@app.get("/api/groups", response_model=GroupsResponse)
async def groups() -> GroupsResponse:
    """Return the currently available Graphiti group identifiers."""

    try:
        return await graphiti_app.list_groups()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface the integration error to the UI.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/history/{session_id}", response_model=HistoryResponse)
async def history(session_id: str) -> HistoryResponse:
    """Load the chat history for one stored session."""

    return await graphiti_app.get_history(session_id)


@app.post("/api/session/reset", response_model=HistoryResponse)
async def reset_session(request: ResetSessionRequest) -> HistoryResponse:
    """Clear the stored history for one chat session."""

    return await graphiti_app.reset_session(request)


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Import free-form text into the local Graphiti graph store."""

    try:
        return await graphiti_app.ingest(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface the integration error to the UI.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using Graphiti retrieval and a short rolling chat history."""

    try:
        return await graphiti_app.chat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface the integration error to the UI.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


__all__ = ["app"]