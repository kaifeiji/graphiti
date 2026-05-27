"""Cross-encoder fallback for Graphiti setups that do not use reranker-based search."""

from __future__ import annotations

from graphiti_core.cross_encoder.client import CrossEncoderClient


class NoOpCrossEncoderClient(CrossEncoderClient):
    """Return passages in their original order with descending placeholder scores."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        del query
        total = len(passages)
        return [
            (passage, float(total - index))
            for index, passage in enumerate(passages)
        ]


__all__ = ["NoOpCrossEncoderClient"]