"""Shared ArcGIS OpenAI-compatible runtime helpers for the Graphiti backend.

This module mirrors the important ArcGIS integration behavior from the main
repository's model factory while keeping the standalone Graphiti project small.
It builds model gateway base URLs from ArcGIS model names and injects the
configured ArcGIS bearer token only for ArcGIS-hosted model gateway calls.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx
from openai import AsyncOpenAI


DEFAULT_ARCGIS_MODEL_HOST = "aimodelsdev.arcgis.com"
_arcgis_async_http_clients: dict[tuple[str, float, str], httpx.AsyncClient] = {}


def has_arcgis_auth(access_token: str | None) -> bool:
    """Return whether ArcGIS model calls have a configured bearer token."""

    return bool((access_token or "").strip())


def build_arcgis_openai_api_base(model_name: str, model_host: str = DEFAULT_ARCGIS_MODEL_HOST) -> str:
    """Build the ArcGIS-hosted OpenAI-compatible base URL for one model name."""

    normalized_model_name = model_name.strip()
    normalized_model_host = model_host.strip().rstrip("/")
    if not normalized_model_name:
        raise ValueError("ArcGIS model name cannot be empty.")
    if not normalized_model_host:
        raise ValueError("ArcGIS model host cannot be empty.")
    return f"https://{normalized_model_host}/{normalized_model_name}/openai/v1"


def _should_inject_arcgis_auth_header(request_url: str, model_host: str) -> bool:
    """Return whether one outbound request targets the configured ArcGIS model host."""

    normalized_model_host = model_host.strip()
    return bool(normalized_model_host) and normalized_model_host in request_url


def _inject_arcgis_auth_header(request: httpx.Request, model_host: str, access_token: str) -> None:
    """Attach the configured ArcGIS bearer token only for model gateway requests."""

    if not _should_inject_arcgis_auth_header(str(request.url), model_host):
        return

    normalized_access_token = access_token.strip()
    if not normalized_access_token:
        return

    request.headers["X-Esri-Authorization"] = f"Bearer {normalized_access_token}"


class ArcGISModelAuth(httpx.Auth):
    """HTTPX auth shim that injects the configured ArcGIS token on model calls."""

    def __init__(self, model_host: str, access_token: str) -> None:
        self._model_host = model_host
        self._access_token = access_token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Attach the configured ArcGIS token and yield the request once."""

        _inject_arcgis_auth_header(request, self._model_host, self._access_token)
        yield request


def _get_or_create_arcgis_async_http_client(
    model_host: str,
    timeout: float,
    access_token: str,
) -> httpx.AsyncClient:
    """Return a shared async HTTPX client that applies ArcGIS token auth."""

    cache_key = (model_host.strip(), timeout, access_token.strip())
    client = _arcgis_async_http_clients.get(cache_key)
    if client is None:
        client = httpx.AsyncClient(
            auth=ArcGISModelAuth(model_host=cache_key[0], access_token=cache_key[2]),
            timeout=timeout,
        )
        _arcgis_async_http_clients[cache_key] = client
    return client


def create_async_arcgis_openai_client(
    base_url: str,
    access_token: str | None,
    model_host: str,
    timeout: float,
) -> AsyncOpenAI:
    """Create one AsyncOpenAI client that supports ArcGIS bearer-token auth."""

    normalized_access_token = (access_token or "").strip()
    return AsyncOpenAI(
        base_url=base_url,
        api_key="none",
        timeout=timeout,
        http_client=_get_or_create_arcgis_async_http_client(
            model_host=model_host,
            timeout=timeout,
            access_token=normalized_access_token,
        ),
    )


async def close_arcgis_async_http_clients() -> None:
    """Close shared async HTTPX clients used by ArcGIS model wrappers."""

    for client in _arcgis_async_http_clients.values():
        await client.aclose()
    _arcgis_async_http_clients.clear()


__all__ = [
    "DEFAULT_ARCGIS_MODEL_HOST",
    "build_arcgis_openai_api_base",
    "close_arcgis_async_http_clients",
    "create_async_arcgis_openai_client",
    "has_arcgis_auth",
]
