"""Thin wrapper around LiteLLM so each writer can use a different backend.

Model specs are LiteLLM model strings (e.g. `ollama_chat/granite4.2:3b`,
`lm_studio/qwen3-8b`, `anthropic/claude-sonnet-5-5`, `openai/gpt-5.6`) or
aliases defined in `models.toml`.
"""

import logging
import os
import re
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import openai
from pydantic import BaseModel, ValidationError

# Global constants
TIMEOUT_DURATION = 120
MAX_ATTEMPTS = 5

litellm.suppress_debug_info = True
# Some models (e.g. OpenAI reasoning models) reject `temperature`; drop it
# rather than failing the whole run.
litellm.drop_params = True

logger = logging.getLogger(name="story_arena")

# Errors that will not go away by trying again
FATAL_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
)


class LLMError(Exception):
    pass


class ResponseTruncatedError(ValueError):
    """The model stopped because it hit its token limit (`max_tokens`)."""


Message = dict[str, str]


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    return {"role": "assistant", "content": content}


@dataclass
class ModelConfig:
    """Settings for one model. `name` is the alias (or the raw spec)."""

    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_model_aliases(path: str | Path) -> dict[str, ModelConfig]:
    """Load `[models.<alias>]` tables from a TOML file. A missing file is fine."""
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("rb") as file:
        data = tomllib.load(file)

    aliases = {}
    for name, settings in data.get("models", {}).items():
        settings = dict(settings)
        if "model" not in settings:
            raise ValueError(f"Model alias '{name}' in {path} is missing 'model'")
        aliases[name] = ModelConfig(
            name=name,
            model=settings.pop("model"),
            api_base=settings.pop("api_base", None),
            api_key_env=settings.pop("api_key_env", None),
            temperature=settings.pop("temperature", None),
            extra=settings,
        )
    return aliases


def resolve_model(spec: str, aliases: dict[str, ModelConfig]) -> ModelConfig:
    """Turn a model spec from a writer file or the command line into a config."""
    spec = spec.strip()
    if spec in aliases:
        return aliases[spec]

    # Backwards compatibility with the old `ollama:<model>` syntax
    if spec.startswith("ollama:"):
        return ModelConfig(
            name=spec, model="ollama_chat/" + spec.removeprefix("ollama:")
        )

    if "/" not in spec:
        raise ValueError(
            f"Unknown model '{spec}'. Use an alias from models.toml or a LiteLLM "
            "model string with a provider prefix, such as 'ollama_chat/granite4.2:3b' "
            "or 'anthropic/claude-sonnet-5-5'."
        )
    return ModelConfig(name=spec, model=spec)


def _strip_code_fences(text: str) -> str:
    """Unwrap a response that is entirely one fenced block (```json ... ```).

    Fences inside the JSON (e.g. code in a story sentence) are left alone.
    """
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text.strip()


class LLM:
    def __init__(
        self,
        config: ModelConfig,
        temperature: float | None = None,
        on_retry: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.on_retry = on_retry
        # an explicit temperature (from --temperature) wins, then the alias's;
        # None means don't send one, so the model's own default applies
        self.temperature = (
            temperature if temperature is not None else config.temperature
        )

    def __str__(self) -> str:
        if self.config.name == self.config.model:
            return self.config.model
        return f"{self.config.name} ({self.config.model})"

    def _kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "timeout": TIMEOUT_DURATION,
            **self.config.extra,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise LLMError(
                    f"Environment variable {self.config.api_key_env} is not set "
                    f"(needed by model '{self.config.name}')"
                )
            kwargs["api_key"] = api_key
        return kwargs

    async def _call(self, messages: list[Message], **kwargs: Any) -> tuple[str, str]:
        """Return the response text and the finish reason."""
        response = await litellm.acompletion(
            messages=messages, **self._kwargs(), **kwargs
        )
        choice = response.choices[0]
        return (choice.message.content or "", choice.finish_reason)

    async def complete(self, messages: list[Message]) -> str:
        """Return a free-form text response."""

        async def attempt() -> str:
            text, _ = await self._call(messages)
            return text

        return await self._with_retries(attempt)

    async def structured[T: BaseModel](
        self, schema: type[T], messages: list[Message]
    ) -> T:
        """Return a response parsed into the given Pydantic schema."""

        async def attempt() -> T:
            text, finish_reason = await self._call(messages, response_format=schema)
            if finish_reason == "length":
                # cut-off JSON can't be parsed; say why instead of a parse error
                raise ResponseTruncatedError(
                    "response was cut off by the token limit; "
                    "raise max_tokens for this model in models.toml"
                )
            return schema.model_validate_json(_strip_code_fences(text))

        return await self._with_retries(attempt)

    async def _with_retries[R](self, fn: Callable[[], Awaitable[R]]) -> R:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await fn()
            except FATAL_ERRORS as e:
                raise LLMError(f"{self} request failed: {e}") from e
            except (openai.APIError, ValidationError, ValueError) as e:
                last_error = e
                msg = (
                    f"{self}: attempt {attempt}/{MAX_ATTEMPTS} failed "
                    f"({type(e).__name__}), trying again"
                )
                logger.warning(f"{msg}: {e}")
                if self.on_retry:
                    self.on_retry(msg)
        raise LLMError(
            f"{self} failed after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error
