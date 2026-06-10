"""Resolve a red-team TARGET model id to a ready backend.

Two routes, chosen by the model id:

  - Self-hosted OSS targets (gpt-oss, Qwen, Llama, Mistral, ...) are served on
    Modal via vLLM (OpenAI-compatible /v1). This is the ban-safe default: we own
    the inference, so only the model license applies — no provider red-teaming
    prohibition, and the target is the bare model with no moderation wrapper.

  - Anything else is treated as a hosted slug and routed through llmcore's
    Router (OpenRouter gateway). NOTE: red-teaming hosted frontier models via a
    first-party API or OpenRouter without written authorization violates ToS and
    risks account termination. Only use that route for authorized targets.

Kept here (not in the shared llmcore catalog) so the redteam project stays
decoupled from ../koverage/core — adding a target is a redteam-local change.
"""

from __future__ import annotations

from llmcore.config import CoreSettings, settings as default_settings
from llmcore.providers.openai_compatible import OpenAICompatibleBackend
from llmcore.providers.router import Router

# Model-id prefixes we serve ourselves on Modal. Matching is case-insensitive.
OSS_TARGET_PREFIXES = (
    "openai/gpt-oss",
    "qwen/",
    "meta-llama/",
    "mistralai/",
    "google/gemma",
    "deepseek-ai/",
)

DEFAULT_TARGET = "openai/gpt-oss-20b"


def is_oss_target(model: str) -> bool:
    """True if `model` should be served by our self-hosted Modal endpoint."""
    m = model.lower()
    return any(m.startswith(p) for p in OSS_TARGET_PREFIXES)


def build_target_backend(
    model: str = DEFAULT_TARGET,
    *,
    settings: CoreSettings | None = None,
) -> OpenAICompatibleBackend:
    """Build the backend for the red-team target `model`.

    OSS targets require MODAL_OSS_URL to be set (the URL printed by
    `modal deploy deploy/modal_gpt_oss.py`). Hosted slugs fall back to the
    llmcore Router.
    """
    settings = settings or default_settings

    if is_oss_target(model):
        url = (settings.modal_oss_url or "").strip()
        if not url:
            raise ValueError(
                f"Target {model!r} is self-hosted but MODAL_OSS_URL is not set.\n"
                "  1. modal deploy deploy/modal_gpt_oss.py\n"
                "  2. put the printed URL in .env as MODAL_OSS_URL=...\n"
                "See .env.example."
            )
        return OpenAICompatibleBackend(
            provider="oss-modal",
            model=model,
            base_url=url.rstrip("/") + "/v1",
            api_key=settings.modal_oss_api_key or "modal",
        )

    # Hosted slug — authorized targets only (see module docstring).
    return Router(settings).backend_for(model)
