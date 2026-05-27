"""Configuration models for the standalone Graphiti backend.

This module centralizes all environment-driven settings for the isolated project.
The goal is to keep startup logic predictable, make the file paths explicit,
and avoid leaking assumptions from the existing repository services.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Typed runtime settings for the Graphiti backend."""

    project_root: Path
    storage_dir: Path
    session_store_path: Path
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8010)
    graph_group_id: str = Field(default="graphiti")
    neo4j_uri: str = Field(default="bolt://127.0.0.1:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str | None = Field(default=None)
    neo4j_database: str = Field(default="neo4j")
    search_limit: int = Field(default=6)
    chunk_size: int = Field(default=1400)
    max_history_messages: int = Field(default=8)
    max_completion_tokens: int = Field(default=1200)
    graphiti_telemetry_enabled: bool = Field(default=False)
    arcgis_model_host: str = Field(default="aimodelsdev.arcgis.com")
    arcgis_timeout_seconds: float = Field(default=60.0)
    arcgis_access_token: str | None = Field(default=None)
    arcgis_chat_model: str = Field(default="gpt-4.1-mini")
    arcgis_small_chat_model: str = Field(default="gpt-4.1-mini")
    arcgis_embedding_model: str = Field(default="text-embedding-3-small")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Build a settings object from the project's local environment files."""

        resolved_root: Path = project_root or Path(__file__).resolve().parents[1]
        env_path: Path = resolved_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        storage_dir: Path = resolved_root / "storage"

        return cls(
            project_root=resolved_root,
            storage_dir=storage_dir,
            session_store_path=storage_dir / "sessions.json",
            host=os.getenv("GRAPHITI_HOST", "0.0.0.0"),
            port=int(os.getenv("GRAPHITI_PORT", "8010")),
            graph_group_id=os.getenv("GRAPHITI_GROUP_ID", "graphiti"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            search_limit=int(os.getenv("GRAPHITI_SEARCH_LIMIT", "6")),
            chunk_size=int(os.getenv("GRAPHITI_CHUNK_SIZE", "1400")),
            max_history_messages=int(os.getenv("GRAPHITI_MAX_HISTORY_MESSAGES", "8")),
            max_completion_tokens=int(os.getenv("GRAPHITI_MAX_COMPLETION_TOKENS", "1200")),
            graphiti_telemetry_enabled=os.getenv(
                "GRAPHITI_TELEMETRY_ENABLED", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            arcgis_model_host=os.getenv("ARCGIS_MODEL_HOST", "aimodelsdev.arcgis.com"),
            arcgis_timeout_seconds=float(os.getenv("ARCGIS_TIMEOUT_SECONDS", "60.0")),
            arcgis_access_token=os.getenv("ARCGIS_ACCESS_TOKEN"),
            arcgis_chat_model=os.getenv("ARCGIS_CHAT_MODEL", "gpt-4.1-mini"),
            arcgis_small_chat_model=os.getenv(
                "ARCGIS_SMALL_CHAT_MODEL", "gpt-4.1-mini"
            ),
            arcgis_embedding_model=os.getenv(
                "ARCGIS_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
        )


__all__ = ["Settings"]
