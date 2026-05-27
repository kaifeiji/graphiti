"""Compatibility wrapper for graphiti_core's Neo4jDriver.

The upstream Neo4jDriver schedules background index creation during __init__.
That interacts poorly with this project's short-lived startup/shutdown checks and
produces noisy retries plus unawaited coroutine warnings on interpreter exit.
This wrapper keeps the same public behavior but leaves index creation under the
caller’s explicit control.
"""

from __future__ import annotations

from neo4j import AsyncGraphDatabase, NotificationMinimumSeverity

from graphiti_core.driver.neo4j.operations.community_edge_ops import Neo4jCommunityEdgeOperations
from graphiti_core.driver.neo4j.operations.community_node_ops import Neo4jCommunityNodeOperations
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.neo4j.operations.entity_node_ops import Neo4jEntityNodeOperations
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neo4j.operations.episodic_edge_ops import Neo4jEpisodicEdgeOperations
from graphiti_core.driver.neo4j.operations.graph_ops import Neo4jGraphMaintenanceOperations
from graphiti_core.driver.neo4j.operations.has_episode_edge_ops import (
    Neo4jHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.next_episode_edge_ops import (
    Neo4jNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.saga_node_ops import Neo4jSagaNodeOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.driver.neo4j_driver import Neo4jDriver


class CompatibleNeo4jDriver(Neo4jDriver):
    """Neo4j driver shim without eager background index scheduling."""

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = "neo4j",
    ):
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or "", password or ""),
            notifications_min_severity=NotificationMinimumSeverity.OFF,
        )
        self._database = database

        self._entity_node_ops = Neo4jEntityNodeOperations()
        self._episode_node_ops = Neo4jEpisodeNodeOperations()
        self._community_node_ops = Neo4jCommunityNodeOperations()
        self._saga_node_ops = Neo4jSagaNodeOperations()
        self._entity_edge_ops = Neo4jEntityEdgeOperations()
        self._episodic_edge_ops = Neo4jEpisodicEdgeOperations()
        self._community_edge_ops = Neo4jCommunityEdgeOperations()
        self._has_episode_edge_ops = Neo4jHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = Neo4jNextEpisodeEdgeOperations()
        self._search_ops = Neo4jSearchOperations()
        self._graph_ops = Neo4jGraphMaintenanceOperations()
        self.aoss_client = None


__all__ = ["CompatibleNeo4jDriver"]