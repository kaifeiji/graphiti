"""Shared Graphiti application layer used by both FastAPI and Chainlit.

This module keeps the retrieval, answer generation, and session persistence flow
in one place so different UIs can reuse the same behavior without drifting.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GroupsResponse,
    HealthResponse,
    HistoryResponse,
    IngestRequest,
    IngestResponse,
    ResetSessionRequest,
)
from app.services.arcgis_client import ArcGISChatClient
from app.services.graphiti_service import GraphitiRAGService
from app.services.session_store import SessionStore


class GraphitiApplication:
    """Owns the long-lived services and the core chat orchestration."""

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.graphiti_service: GraphitiRAGService = GraphitiRAGService(settings=settings)
        self.session_store: SessionStore = SessionStore(storage_path=settings.session_store_path)
        self.arcgis_client: ArcGISChatClient = ArcGISChatClient(settings=settings)
        self._startup_lock: asyncio.Lock = asyncio.Lock()
        self._started: bool = False

    async def startup(self) -> None:
        """Initialize service singletons once for the current process."""

        async with self._startup_lock:
            if self._started:
                return

            await self.session_store.startup()
            await self.graphiti_service.startup()
            self._started = True

    async def shutdown(self) -> None:
        """Release process-wide resources when the host exits."""

        async with self._startup_lock:
            if not self._started:
                return

            await self.graphiti_service.shutdown()
            self._started = False

    def health(self) -> HealthResponse:
        """Return a stable readiness payload for any UI surface."""

        return HealthResponse(
            status="ok",
            graphiti_ready=self.graphiti_service.is_ready(),
            arcgis_configured=self.arcgis_client.is_configured(),
            startup_error=self.graphiti_service.startup_error(),
            graph_database_uri=self.settings.neo4j_uri,
            graph_database_name=self.settings.neo4j_database,
            session_store_path=str(self.settings.session_store_path),
        )

    async def list_groups(self) -> GroupsResponse:
        """Return available Graphiti groups for group-aware chat clients."""

        self._require_graphiti_ready()
        return GroupsResponse(
            default_group=self.settings.graph_group_id,
            groups=await self.graphiti_service.list_groups(),
        )

    async def get_history(self, session_id: str) -> HistoryResponse:
        """Return persisted history for one session id."""

        return HistoryResponse(
            session_id=session_id,
            history=await self.session_store.get_history(session_id),
        )

    async def reset_session(self, request: ResetSessionRequest) -> HistoryResponse:
        """Clear stored messages for one session id."""

        await self.session_store.clear_history(request.session_id)
        return HistoryResponse(session_id=request.session_id, history=[])

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Ingest free-form text into the configured default group."""

        self._require_graphiti_ready()
        episodes_added = await self.graphiti_service.ingest_text(
            title=request.title,
            content=request.content,
            source_description=request.source_description,
        )
        return IngestResponse(title=request.title, episodes_added=episodes_added)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Answer one user message and persist the resulting chat turn."""

        self._require_graphiti_ready()

        history_before: list[ChatMessage] = await self.session_store.get_history(request.session_id)
        retrieved_context = await self.graphiti_service.retrieve_context(
            request.message,
            group_id=request.group,
        )
        answer: str = await self.arcgis_client.answer_question(
            question=request.message,
            history=history_before,
            context_text=retrieved_context.context_text,
        )

        await self.session_store.append_message(
            request.session_id,
            ChatMessage(role="user", content=request.message),
        )
        updated_history = await self.session_store.append_message(
            request.session_id,
            ChatMessage(
                role="assistant",
                content=answer,
                context=retrieved_context.items,
            ),
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            history=updated_history,
            context=retrieved_context.items,
        )

    def _require_graphiti_ready(self) -> None:
        """Raise a consistent error when retrieval is requested before startup succeeds."""

        if self.graphiti_service.is_ready():
            return
        raise RuntimeError(
            self.graphiti_service.startup_error() or "Graphiti is still starting."
        )


__all__ = ["GraphitiApplication"]