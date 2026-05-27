"""Request, response, and transport models for the Graphiti backend.

These models define a very small HTTP contract for the standalone project. They are
kept intentionally explicit so the API stays predictable across multiple UI layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


ChatRole = Literal["system", "user", "assistant"]


class ContextItem(BaseModel):
    """Represents one context fragment extracted from Graphiti search results."""

    kind: Literal["fact", "entity", "episode", "community"]
    title: str
    content: str


class ChatMessage(BaseModel):
    """Represents a single chat turn stored by the session layer."""

    role: ChatRole
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: list[ContextItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Summarizes backend readiness and local storage status."""

    status: str
    graphiti_ready: bool
    arcgis_configured: bool
    startup_error: str | None = None
    graph_database_uri: str
    graph_database_name: str
    session_store_path: str


class GroupsResponse(BaseModel):
    """Lists the currently available Graphiti group identifiers."""

    default_group: str
    groups: list[str]


class HistoryResponse(BaseModel):
    """Returns the full in-memory history for a single session."""

    session_id: str
    history: list[ChatMessage]


class IngestRequest(BaseModel):
    """Payload for adding free-form text into the local Graphiti graph."""

    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1)
    source_description: str = Field(default="Manual text import", max_length=200)


class IngestResponse(BaseModel):
    """Reports how many graph episodes were created from the import request."""

    title: str
    episodes_added: int


class ChatRequest(BaseModel):
    """Payload for asking a question against the Graphiti-backed context graph."""

    session_id: str = Field(default="default", min_length=1, max_length=80)
    message: str = Field(min_length=1)
    group: str | None = Field(default=None, max_length=200)


class ChatResponse(BaseModel):
    """Returns the answer, updated history, and retrieved context snippets."""

    session_id: str
    answer: str
    history: list[ChatMessage]
    context: list[ContextItem]


class ResetSessionRequest(BaseModel):
    """Payload for clearing the chat history of a single session."""

    session_id: str = Field(default="default", min_length=1, max_length=80)


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ContextItem",
    "GroupsResponse",
    "HealthResponse",
    "HistoryResponse",
    "IngestRequest",
    "IngestResponse",
    "ResetSessionRequest",
]
