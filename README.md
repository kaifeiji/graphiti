# Graphiti

This directory contains a fully isolated Graphiti RAG project. It does not reuse the repository's existing backend, frontend, build scripts, or deployment flow.

## What it includes

- A minimal FastAPI backend under graphiti/app
- A Chainlit chat frontend under graphiti/chainlit_app.py
- A local Graphiti service backed by Neo4j Desktop
- A cross-platform Python launcher under graphiti/run.py
- A uv-managed project configuration in graphiti/pyproject.toml

## Behavior

1. Start the Chainlit app and choose a Neo4j-backed Graphiti group from chat settings.
2. Ask follow-up questions in the chat thread against that group.
3. Inspect the retrieved Graphiti context attached to each assistant answer.
4. Import new text through the backend API or scripts when you need to add data.

## ArcGIS LLM assumptions

This project assumes your ArcGIS-hosted LLM service exposes an OpenAI-compatible API surface for:

- chat completions
- embeddings

The backend builds ArcGIS model gateway URLs from ARCGIS_MODEL_HOST plus the configured model names, following the same pattern used in the repository's main llm_factory.

The backend uses one ArcGIS authentication mode:

- ARCGIS_ACCESS_TOKEN from graphiti/.env, which the backend forwards to ArcGIS model calls through X-Esri-Authorization

If your ArcGIS environment uses a different protocol, adapt graphiti/app/services/arcgis_client.py, graphiti/app/services/graphiti_runtime.py, and graphiti/app/services/graphiti_service.py.

## Quick start

1. Copy graphiti/.env.example to graphiti/.env.
2. Fill in your ArcGIS token, model host, chat model, embedding model, and Neo4j connection settings.
3. Install uv.
4. Run `uv sync` in graphiti.
5. Run `uv run python run.py`.
4. Open http://127.0.0.1:8010 on the host machine, or use the machine's LAN IP from another device.

## Manual start

### Chainlit app

```
uv sync
uv run python run.py
```

### Backend API only

```
uv sync
uv run python run.py api
```

### Evaluation

```
uv sync --extra eval
uv run python evaluation\run_hotpotqa_beir.py --max-queries 20 --negative-docs 200
```

## API overview

- GET /api/health
- GET /api/groups
- GET /api/history/{session_id}
- POST /api/ingest
- POST /api/chat
- POST /api/session/reset

`POST /api/chat` now accepts an optional `group` field. When provided, retrieval is restricted to that Graphiti group. When omitted, the backend falls back to the configured default group and may still try recent evaluation groups when the default has no hits.

The default UI is now Chainlit. To ingest data, call `POST /api/ingest` directly or use the evaluation/runtime scripts.

## Local storage

- Neo4j stores the graph data in your local Neo4j Desktop instance.
- graphiti/storage/sessions.json: the persisted UI chat history

## Notes

- The default UI is now Chainlit and binds to 0.0.0.0 by default for LAN access.
- The FastAPI app is still available for API use and automation, but it is no longer the primary chat UI.
- This workspace now uses uv for dependency management; `requirements.txt`, `run.ps1`, and `run.bat` have been removed.
- The backend can initialize without a valid ArcGIS token, but actual model calls require ARCGIS_ACCESS_TOKEN to be set in graphiti/.env.
- The search pipeline uses the Graphiti RRF recipe instead of the cross-encoder recipe so the project stays compatible with a broader range of OpenAI-compatible providers.
