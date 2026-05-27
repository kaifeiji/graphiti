"""Minimal persistent chat session storage for the Graphiti backend.

This store keeps the UI conversation state separate from Graphiti itself. That
lets the project show recent chat history even though the graph is primarily used
for document ingestion and retrieval rather than full conversation memory.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.models import ChatMessage


class SessionStore:
    """Stores session history in a small JSON file under the storage folder."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path: Path = storage_path
        self._lock: asyncio.Lock = asyncio.Lock()
        self._sessions: dict[str, list[ChatMessage]] = {}

    async def startup(self) -> None:
        """Create the storage file when missing and load any saved sessions."""

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._storage_path.exists():
            self._storage_path.write_text("{}", encoding="utf-8")

        raw_payload: str = self._storage_path.read_text(encoding="utf-8").strip() or "{}"
        loaded_data: dict[str, list[dict[str, str]]] = json.loads(raw_payload)
        self._sessions = {
            session_id: [ChatMessage.model_validate(message) for message in messages]
            for session_id, messages in loaded_data.items()
        }

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of the stored history so callers cannot mutate shared state."""

        async with self._lock:
            return [message.model_copy(deep=True) for message in self._sessions.get(session_id, [])]

    async def append_message(self, session_id: str, message: ChatMessage) -> list[ChatMessage]:
        """Append one message to a session and persist the updated history to disk."""

        async with self._lock:
            self._sessions.setdefault(session_id, []).append(message)
            self._persist()
            return [item.model_copy(deep=True) for item in self._sessions[session_id]]

    async def clear_history(self, session_id: str) -> None:
        """Delete a session history while leaving other sessions intact."""

        async with self._lock:
            self._sessions.pop(session_id, None)
            self._persist()

    def _persist(self) -> None:
        """Synchronously write the full session map to disk for simplicity."""

        serialized: dict[str, list[dict[str, str]]] = {
            session_id: [message.model_dump(mode="json") for message in messages]
            for session_id, messages in self._sessions.items()
        }
        self._storage_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["SessionStore"]
