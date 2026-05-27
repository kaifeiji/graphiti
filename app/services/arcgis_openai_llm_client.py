"""ArcGIS-compatible Graphiti LLM client helpers.

This project uses ArcGIS-hosted models through an OpenAI-compatible API, but
some newer model families require max_completion_tokens instead of max_tokens.
This local adapter keeps Graphiti compatible without patching third-party code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL, OpenAIGenericClient
from graphiti_core.prompts.models import Message
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ArcGISOpenAIGenericClient(OpenAIGenericClient):
    """Use max_completion_tokens so ArcGIS-hosted GPT-5 family models work."""

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role == "user":
                openai_messages.append({"role": "user", "content": message.content})
            elif message.role == "system":
                openai_messages.append({"role": "system", "content": message.content})

        try:
            response_format: dict[str, Any] = {"type": "json_object"}
            if response_model is not None:
                schema_name = getattr(response_model, "__name__", "structured_response")
                json_schema = response_model.model_json_schema()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                    },
                }

            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_completion_tokens=max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ""
            return json.loads(result)
        except openai.RateLimitError as error:
            raise RateLimitError from error
        except Exception as error:
            logger.error(f"Error in generating LLM response: {error}")
            raise


__all__ = ["ArcGISOpenAIGenericClient"]