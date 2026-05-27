"""ArcGIS-compatible chat client for the Graphiti backend.

This client assumes the ArcGIS-hosted LLM endpoint exposes an OpenAI-compatible
chat completions interface. The adapter is intentionally small: it only builds
the prompt needed by the project and delegates transport details to the OpenAI SDK.
"""

from __future__ import annotations

from app.config import Settings
from app.models import ChatMessage
from app.services.arcgis_runtime import (
    build_arcgis_openai_api_base,
    create_async_arcgis_openai_client,
    has_arcgis_auth,
)


class ArcGISChatClient:
    """Wraps a single OpenAI-compatible async client for answer generation."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        resolved_base_url = build_arcgis_openai_api_base(
            model_name=settings.arcgis_chat_model,
            model_host=settings.arcgis_model_host,
        )
        self._client = create_async_arcgis_openai_client(
            base_url=resolved_base_url,
            access_token=settings.arcgis_access_token,
            model_host=settings.arcgis_model_host,
            timeout=settings.arcgis_timeout_seconds,
        )

    def is_configured(self) -> bool:
        """Expose whether the current settings are enough to call the model."""

        return has_arcgis_auth(self._settings.arcgis_access_token)

    async def answer_question(
        self,
        question: str,
        history: list[ChatMessage],
        context_text: str,
    ) -> str:
        """Generate a grounded answer using retrieved graph context and recent history."""

        if not has_arcgis_auth(self._settings.arcgis_access_token):
            raise RuntimeError(
                "ArcGIS LLM authentication is not configured. Set ARCGIS_ACCESS_TOKEN in graphiti/.env."
            )

        recent_history: list[ChatMessage] = history[-self._settings.max_history_messages :]
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are the answering layer for a Graphiti-powered RAG system. "
                    "Use the retrieved graph context first, then use recent chat history for continuity. "
                    "If the graph context is empty or insufficient, say so plainly instead of inventing facts."
                ),
            },
            {
                "role": "system",
                "content": f"Retrieved graph context:\n{context_text}",
            },
        ]

        # The project keeps only a short rolling conversation window so prompts stay small and predictable.
        for message in recent_history:
            messages.append({"role": message.role, "content": message.content})

        messages.append({"role": "user", "content": question})

        completion = await self._client.chat.completions.create(
            model=self._settings.arcgis_chat_model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=self._settings.max_completion_tokens,
        )
        answer: str = completion.choices[0].message.content or "I could not generate a response."
        return answer.strip()


__all__ = ["ArcGISChatClient"]
