"""Chainlit frontend for the standalone Graphiti chat experience."""

from __future__ import annotations

from uuid import uuid4

import chainlit as cl
from chainlit.input_widget import Select

from app.application import GraphitiApplication
from app.config import Settings
from app.models import ChatRequest, ContextItem


settings = Settings.from_env()
graphiti_app = GraphitiApplication(settings)
GROUP_SETTING_ID = "group"


def _format_context(items: list[ContextItem]) -> str:
    """Render retrieved context into compact markdown for the Chainlit side panel."""

    if not items:
        return "No relevant graph context was found in the local Graphiti store."

    sections: list[str] = []
    for item in items:
        sections.append(
            "\n".join(
                [
                    f"### {item.kind.title()}: {item.title}",
                    item.content,
                ]
            )
        )
    return "\n\n".join(sections)


async def _send_group_settings(selected_group: str | None = None) -> None:
    """Expose the available Graphiti groups in Chainlit chat settings."""

    groups_response = await graphiti_app.list_groups()
    groups = groups_response.groups or [groups_response.default_group]
    resolved_group = (selected_group or "").strip() or groups_response.default_group
    if resolved_group not in groups:
        resolved_group = groups_response.default_group

    cl.user_session.set("group", resolved_group)
    await cl.ChatSettings(
        [
            Select(
                id=GROUP_SETTING_ID,
                label="Graph group",
                values=groups,
                initial_index=groups.index(resolved_group),
            )
        ]
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize shared services and open a new stored Graphiti session."""

    await graphiti_app.startup()
    session_id = str(uuid4())
    cl.user_session.set("session_id", session_id)
    await _send_group_settings(settings.graph_group_id)

    health = graphiti_app.health()
    status_lines = [
        f"Connected to Graphiti group `{settings.graph_group_id}`.",
        f"Neo4j: `{health.graph_database_name}` on `{health.graph_database_uri}`.",
    ]
    if not health.arcgis_configured:
        status_lines.append(
            "ArcGIS LLM authentication is not configured yet. Set `ARCGIS_ACCESS_TOKEN` before asking questions."
        )

    await cl.Message(content="\n\n".join(status_lines)).send()


@cl.on_settings_update
async def on_settings_update(updated_settings: dict) -> None:
    """Persist the active group selected from the Chainlit settings panel."""

    selected_group = (updated_settings.get(GROUP_SETTING_ID) or "").strip()
    if selected_group:
        cl.user_session.set("group", selected_group)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer one incoming Chainlit user message using Graphiti retrieval."""

    await graphiti_app.startup()
    session_id = cl.user_session.get("session_id") or str(uuid4())
    cl.user_session.set("session_id", session_id)
    group = cl.user_session.get("group")

    response = await graphiti_app.chat(
        ChatRequest(
            session_id=session_id,
            message=message.content,
            group=group,
        )
    )
    elements = [
        cl.Text(
            name="Retrieved context",
            content=_format_context(response.context),
            display="side",
        )
    ]
    await cl.Message(content=response.answer, elements=elements).send()