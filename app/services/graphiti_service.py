"""Graphiti-backed retrieval service for the standalone backend.

This service owns the local Neo4j-backed graph, document ingestion, and retrieval
formatting. It intentionally keeps the public API small so the rest of the project
can treat Graphiti as a focused RAG engine rather than a general graph platform.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_helpers import search_results_to_context_string
from graphiti_core.utils.bulk_utils import RawEpisode

from app.config import Settings
from app.models import ContextItem
from app.services.arcgis_openai_llm_client import ArcGISOpenAIGenericClient
from app.services.compatible_neo4j_driver import CompatibleNeo4jDriver
from app.services.noop_cross_encoder import NoOpCrossEncoderClient
from app.services.arcgis_runtime import (
    build_arcgis_openai_api_base,
    close_arcgis_async_http_clients,
    create_async_arcgis_openai_client,
)


MAX_FACT_SOURCE_EPISODES = 4
MAX_DIRECT_SOURCE_EPISODES = 2
MAX_FACT_SOURCE_EDGES = 8
MAX_CONTEXT_FACTS = 8
MAX_CONTEXT_ENTITIES = 8
MAX_CONTEXT_EPISODES = 4
MAX_SOURCE_EXCERPT_CHARS = 3200
MAX_SOURCE_EXCERPT_LENGTH = 900
SOURCE_EXCERPT_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "before",
    "both",
    "by",
    "did",
    "do",
    "does",
    "first",
    "for",
    "from",
    "how",
    "in",
    "is",
    "last",
    "later",
    "name",
    "of",
    "on",
    "or",
    "other",
    "same",
    "than",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "year",
}


@dataclass(slots=True)
class RetrievedContext:
    """Bundles Graphiti retrieval output for both the UI and the answering layer."""

    context_text: str
    items: list[ContextItem]


class GraphitiRAGService:
    """Initializes Graphiti once and exposes minimal ingest and search methods."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._graphiti: Graphiti | None = None
        self._startup_error: str | None = self._missing_config_message()

    async def startup(self) -> None:
        """Prepare runtime directories and construct the Neo4j-backed Graphiti instance."""

        self._settings.storage_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = (
            "true" if self._settings.graphiti_telemetry_enabled else "false"
        )
        if not self.can_initialize():
            self._startup_error = self._missing_config_message()
            return
        self._graphiti = self._create_graphiti()
        await self._graphiti.driver.health_check()
        await self._graphiti.build_indices_and_constraints(delete_existing=False)
        self._startup_error = None

    async def shutdown(self) -> None:
        """Release driver resources when the ASGI server stops."""

        if self._graphiti is not None:
            await self._graphiti.driver.close()
        await close_arcgis_async_http_clients()

    def is_ready(self) -> bool:
        """Report whether the Graphiti instance has been initialized."""

        return self._graphiti is not None

    def can_initialize(self) -> bool:
        """Return whether the current configuration is sufficient to construct Graphiti."""

        return bool(self._settings.arcgis_chat_model.strip() and self._settings.arcgis_embedding_model.strip())

    def startup_error(self) -> str | None:
        """Expose the most recent startup issue for diagnostics and the health endpoint."""

        return self._startup_error

    def _missing_config_message(self) -> str | None:
        """Return a stable message when required ArcGIS configuration is absent."""

        if self.can_initialize():
            return None
        return "Graphiti startup skipped because ARCGIS_CHAT_MODEL or ARCGIS_EMBEDDING_MODEL is missing."

    async def ingest_text(self, title: str, content: str, source_description: str) -> int:
        """Split imported text into stable chunks and ingest them into Graphiti."""

        graphiti: Graphiti = self._require_graphiti()
        chunks: list[str] = self._chunk_text(content)

        # Each chunk becomes one text episode so Graphiti can extract and persist graph facts incrementally.
        for index, chunk in enumerate(chunks, start=1):
            await graphiti.add_episode(
                name=f"{title} #{index}",
                episode_body=chunk,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id=self._settings.graph_group_id,
            )

        return len(chunks)

    async def ingest_text_bulk(
        self,
        documents: list[tuple[str, str, str]],
    ) -> int:
        """Bulk-ingest multiple texts through Graphiti's batch episode path."""

        graphiti: Graphiti = self._require_graphiti()
        reference_time = datetime.now(timezone.utc)
        raw_episodes: list[RawEpisode] = []

        for title, content, source_description in documents:
            for index, chunk in enumerate(self._chunk_text(content), start=1):
                raw_episodes.append(
                    RawEpisode(
                        name=f"{title} #{index}",
                        content=chunk,
                        source_description=source_description,
                        source=EpisodeType.text,
                        reference_time=reference_time,
                    )
                )

        if not raw_episodes:
            return 0

        await graphiti.add_episode_bulk(
            raw_episodes,
            group_id=self._settings.graph_group_id,
        )
        return len(raw_episodes)

    async def retrieve_context(
        self,
        query: str,
        group_id: str | None = None,
    ) -> RetrievedContext:
        """Run Graphiti retrieval and convert results into prompt text plus UI snippets."""

        graphiti: Graphiti = self._require_graphiti()
        search_config = copy.deepcopy(COMBINED_HYBRID_SEARCH_RRF)
        search_config.limit = self._settings.search_limit
        requested_group_id = self._normalize_group_id(group_id)

        results = await graphiti.search_(
            query=query,
            config=search_config,
            group_ids=[requested_group_id or self._settings.graph_group_id],
        )
        if requested_group_id is None and self._is_empty_search_result(results):
            for fallback_group_id in self._iter_eval_fallback_group_ids():
                results = await graphiti.search_(
                    query=query,
                    config=search_config,
                    group_ids=[fallback_group_id],
                )
                if not self._is_empty_search_result(results):
                    break

        items: list[ContextItem] = self._build_context_items(results)
        context_text: str = (
            await self._build_context_text(
                query,
                results,
                requested_group_id or self._settings.graph_group_id,
            )
            if items
            else "No relevant graph context was found in the local Graphiti store."
        )
        return RetrievedContext(context_text=context_text, items=items)

    async def list_groups(self) -> list[str]:
        """List distinct Graphiti group ids currently present in Neo4j."""

        graphiti: Graphiti = self._require_graphiti()
        result = await graphiti.driver.execute_query(
            """
            CALL {
                MATCH (episode:Episodic)
                WHERE episode.group_id IS NOT NULL AND episode.group_id <> ''
                RETURN episode.group_id AS group_id
                UNION
                MATCH (entity:Entity)
                WHERE entity.group_id IS NOT NULL AND entity.group_id <> ''
                RETURN entity.group_id AS group_id
                UNION
                MATCH (community:Community)
                WHERE community.group_id IS NOT NULL AND community.group_id <> ''
                RETURN community.group_id AS group_id
                UNION
                MATCH (saga:Saga)
                WHERE saga.group_id IS NOT NULL AND saga.group_id <> ''
                RETURN saga.group_id AS group_id
            }
            RETURN DISTINCT group_id
            """
        )
        discovered_groups = {
            group_id.strip()
            for record in result.records
            if isinstance((group_id := record.get("group_id")), str) and group_id.strip()
        }

        discovered_groups.add(self._settings.graph_group_id)
        fallback_groups = set(self._iter_eval_fallback_group_ids())
        discovered_groups.update(fallback_groups)

        ordered_groups = sorted(discovered_groups)
        if self._settings.graph_group_id in discovered_groups:
            ordered_groups.remove(self._settings.graph_group_id)
            ordered_groups.insert(0, self._settings.graph_group_id)
        return ordered_groups

    def _iter_eval_fallback_group_ids(self) -> list[str]:
        """Return recent evaluation group ids so the app can reuse local benchmark ingests."""

        marker_paths: list[Path] = []
        for runtime_root in self._candidate_eval_runtime_roots():
            marker_paths.extend(runtime_root.glob("**/graphiti_eval_ingested.marker"))

        if not marker_paths:
            return []

        marker_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        fallback_group_ids: list[str] = []
        seen_group_ids: set[str] = {self._settings.graph_group_id}
        for marker_path in marker_paths:
            group_id = marker_path.read_text(encoding="utf-8").strip()
            if not group_id or group_id in seen_group_ids:
                continue
            seen_group_ids.add(group_id)
            fallback_group_ids.append(group_id)

        return fallback_group_ids

    def _candidate_eval_runtime_roots(self) -> list[Path]:
        """Return the evaluation runtime directory for the root-level repo layout."""

        candidates = [self._settings.project_root / "evaluation" / "runtime"]
        return [candidate for candidate in candidates if candidate.exists()]

    def _normalize_group_id(self, group_id: str | None) -> str | None:
        """Normalize optional group ids coming from the API layer."""

        if group_id is None:
            return None
        normalized_group_id = group_id.strip()
        return normalized_group_id or None

    def _is_empty_search_result(self, results: object) -> bool:
        """Treat Graphiti search output as empty only when every result bucket is empty."""

        return not any(
            getattr(results, field_name, [])
            for field_name in ("edges", "nodes", "episodes", "communities")
        )

    def _create_graphiti(self) -> Graphiti:
        """Construct a Graphiti instance configured for Neo4j persistence."""

        chat_base_url = build_arcgis_openai_api_base(
            model_name=self._settings.arcgis_chat_model,
            model_host=self._settings.arcgis_model_host,
        )
        embedding_base_url = build_arcgis_openai_api_base(
            model_name=self._settings.arcgis_embedding_model,
            model_host=self._settings.arcgis_model_host,
        )
        chat_async_client = create_async_arcgis_openai_client(
            base_url=chat_base_url,
            access_token=self._settings.arcgis_access_token,
            model_host=self._settings.arcgis_model_host,
            timeout=self._settings.arcgis_timeout_seconds,
        )
        embedding_async_client = create_async_arcgis_openai_client(
            base_url=embedding_base_url,
            access_token=self._settings.arcgis_access_token,
            model_host=self._settings.arcgis_model_host,
            timeout=self._settings.arcgis_timeout_seconds,
        )
        llm_config = LLMConfig(
            api_key="none",
            model=self._settings.arcgis_chat_model,
            small_model=self._settings.arcgis_small_chat_model,
            base_url=chat_base_url,
            temperature=0,
            max_tokens=self._settings.max_completion_tokens,
        )
        llm_client = ArcGISOpenAIGenericClient(
            config=llm_config,
            client=chat_async_client,
            max_tokens=self._settings.max_completion_tokens,
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="none",
                base_url=embedding_base_url,
                embedding_model=self._settings.arcgis_embedding_model,
            ),
            client=embedding_async_client,
        )
        return Graphiti(
            graph_driver=CompatibleNeo4jDriver(
                uri=self._settings.neo4j_uri,
                user=self._settings.neo4j_user,
                password=self._settings.neo4j_password,
                database=self._settings.neo4j_database,
            ),
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=NoOpCrossEncoderClient(),
            max_coroutines=4,
        )

    def _require_graphiti(self) -> Graphiti:
        """Fail fast when a route is called before startup completes."""

        if self._graphiti is None:
            raise RuntimeError("Graphiti is not initialized yet.")
        return self._graphiti

    def _chunk_text(self, content: str) -> list[str]:
        """Create stable text chunks without introducing another external chunker."""

        normalized_blocks: list[str] = [
            block.strip() for block in content.replace("\r\n", "\n").split("\n\n") if block.strip()
        ]
        if not normalized_blocks:
            return [content.strip()]

        chunks: list[str] = []
        current_chunk: str = ""
        for block in normalized_blocks:
            candidate: str = block if not current_chunk else f"{current_chunk}\n\n{block}"
            if len(candidate) <= self._settings.chunk_size:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk)
            if len(block) <= self._settings.chunk_size:
                current_chunk = block
                continue

            # Very large paragraphs are sliced deterministically so repeated imports stay stable.
            for start_index in range(0, len(block), self._settings.chunk_size):
                end_index: int = start_index + self._settings.chunk_size
                chunks.append(block[start_index:end_index])
            current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)
        return chunks or [content.strip()]

    def _build_context_items(self, results: object) -> list[ContextItem]:
        """Flatten Graphiti search results into a compact structure for the UI."""

        items: list[ContextItem] = []

        for edge in getattr(results, "edges", []):
            items.append(
                ContextItem(
                    kind="fact",
                    title=getattr(edge, "name", None) or "Fact",
                    content=getattr(edge, "fact", ""),
                )
            )
        for node in getattr(results, "nodes", []):
            items.append(
                ContextItem(
                    kind="entity",
                    title=getattr(node, "name", "Entity"),
                    content=getattr(node, "summary", ""),
                )
            )
        for episode in getattr(results, "episodes", []):
            items.append(
                ContextItem(
                    kind="episode",
                    title=getattr(episode, "source_description", "Episode"),
                    content=getattr(episode, "content", ""),
                )
            )
        for community in getattr(results, "communities", []):
            items.append(
                ContextItem(
                    kind="community",
                    title=getattr(community, "name", "Community"),
                    content=getattr(community, "summary", ""),
                )
            )

        return [item for item in items if item.content][: self._settings.search_limit]

    async def _build_context_text(self, query: str, results: object, group_id: str) -> str:
        """Render Graphiti results plus supporting source excerpts for fact grounding."""

        base_context_results = self._slice_results_for_context(results)
        base_context: str = search_results_to_context_string(base_context_results)
        supporting_excerpts = await self._collect_fact_source_excerpts(query, results, group_id)
        if not supporting_excerpts:
            return base_context

        return (
            f"{base_context}\n\n"
            "Use the EPISODES section above as the original text for directly retrieved episodes. "
            "The following source excerpts are the original episode texts that support retrieved facts.\n"
            "<SOURCE_EXCERPTS>\n"
            f"{supporting_excerpts}\n"
            "</SOURCE_EXCERPTS>"
        )

    def _slice_results_for_context(self, results: object) -> object:
        """Reduce LLM context noise by trimming each Graphiti result bucket to its top items."""

        return SimpleNamespace(
            edges=list(getattr(results, "edges", []))[:MAX_CONTEXT_FACTS],
            nodes=list(getattr(results, "nodes", []))[:MAX_CONTEXT_ENTITIES],
            episodes=list(getattr(results, "episodes", []))[:MAX_CONTEXT_EPISODES],
            communities=list(getattr(results, "communities", [])),
        )

    async def _collect_fact_source_excerpts(self, query: str, results: object, group_id: str) -> str:
        """Attach original episode text for fact results that expose episode provenance."""

        direct_episode_uuids: set[str] = set()
        for episode in getattr(results, "episodes", [])[:MAX_DIRECT_SOURCE_EPISODES]:
            episode_uuid = str(getattr(episode, "uuid", "")).strip()
            if not episode_uuid:
                continue
            direct_episode_uuids.add(episode_uuid)

        query_terms = self._extract_source_excerpt_query_terms(query)
        supporting_facts_by_episode_uuid: dict[str, list[str]] = {}
        supporting_episode_scores: dict[str, tuple[int, int, int]] = {}

        for edge_rank, edge in enumerate(getattr(results, "edges", [])[:MAX_FACT_SOURCE_EDGES], start=1):
            fact_text = str(getattr(edge, "fact", "")).strip()
            episode_uuids = [
                str(episode_uuid).strip()
                for episode_uuid in getattr(edge, "episodes", []) or []
                if str(episode_uuid).strip()
            ]
            if not fact_text or not episode_uuids:
                continue

            overlap_count = 0
            if query_terms:
                overlap_count = len(query_terms & self._extract_source_excerpt_query_terms(fact_text))

            for episode_uuid in episode_uuids:
                if episode_uuid in direct_episode_uuids:
                    continue
                supporting_facts_by_episode_uuid.setdefault(episode_uuid, []).append(fact_text)
                previous_overlap_count, previous_fact_count, previous_best_rank = supporting_episode_scores.get(
                    episode_uuid,
                    (0, 0, edge_rank),
                )
                supporting_episode_scores[episode_uuid] = (
                    previous_overlap_count + overlap_count,
                    previous_fact_count + 1,
                    min(previous_best_rank, edge_rank),
                )

        if not supporting_facts_by_episode_uuid:
            return ""

        ranked_episode_uuids = sorted(
            supporting_facts_by_episode_uuid,
            key=lambda episode_uuid: (
                -supporting_episode_scores[episode_uuid][0],
                -supporting_episode_scores[episode_uuid][1],
                supporting_episode_scores[episode_uuid][2],
            ),
        )[:MAX_FACT_SOURCE_EPISODES]

        supporting_episodes = await self._load_episodes_by_uuid(
            ranked_episode_uuids,
            group_id,
        )

        rendered_excerpts: list[str] = []
        used_chars = 0
        for episode_uuid in ranked_episode_uuids:
            facts = supporting_facts_by_episode_uuid[episode_uuid]
            episode = supporting_episodes.get(episode_uuid)
            if not episode:
                continue

            fact_summary = "; ".join(dict.fromkeys(facts[:2]))
            excerpt = self._truncate_source_excerpt(episode["content"])
            rendered = "\n".join(
                [
                    f"- source_description: {episode['source_description']}",
                    f"  supports_fact: {fact_summary}",
                    f"  text: {excerpt}",
                ]
            )
            projected_chars = used_chars + len(rendered)
            if rendered_excerpts and projected_chars > MAX_SOURCE_EXCERPT_CHARS:
                break
            rendered_excerpts.append(rendered)
            used_chars = projected_chars

        return "\n\n".join(rendered_excerpts)

    async def _load_episodes_by_uuid(
        self,
        episode_uuids: list[str],
        group_id: str,
    ) -> dict[str, dict[str, str]]:
        """Load episode source text for a bounded set of episode uuids."""

        unique_episode_uuids = list(dict.fromkeys(episode_uuids))
        if not unique_episode_uuids:
            return {}

        graphiti: Graphiti = self._require_graphiti()
        result = await graphiti.driver.execute_query(
            """
            MATCH (episode:Episodic {group_id: $group_id})
            WHERE episode.uuid IN $episode_uuids
            RETURN episode.uuid AS uuid,
                   episode.source_description AS source_description,
                   episode.content AS content
            """,
            params={
                "group_id": group_id,
                "episode_uuids": unique_episode_uuids,
            },
        )
        return {
            str(record["uuid"]): {
                "source_description": str(record["source_description"] or "Episode"),
                "content": str(record["content"] or "").strip(),
            }
            for record in result.records
            if record.get("uuid") and str(record.get("content") or "").strip()
        }

    def _truncate_source_excerpt(self, content: str) -> str:
        """Trim long source excerpts so grounding stays within a predictable token budget."""

        normalized_content = content.strip()
        if len(normalized_content) <= MAX_SOURCE_EXCERPT_LENGTH:
            return normalized_content
        return f"{normalized_content[: MAX_SOURCE_EXCERPT_LENGTH - 3].rstrip()}..."

    def _extract_source_excerpt_query_terms(self, text: str) -> set[str]:
        """Extract stable query terms so excerpt budget is spent on query-relevant facts."""

        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 3 and token not in SOURCE_EXCERPT_QUERY_STOPWORDS
        }


__all__ = ["GraphitiRAGService", "RetrievedContext"]
