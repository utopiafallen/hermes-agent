"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False only on Ollama URLs (/api/chat and proxies)
    + chat_template_kwargs.reasoning_effort = "none"
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
    + chat_template_kwargs.reasoning_effort = <effort>
    (vLLM/SGLang serving Qwen3-style templates read the effort from
    chat_template_kwargs, not the top-level field — emit both)
"""

from typing import Any
from urllib.parse import urlparse

from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


def _looks_like_ollama_endpoint(base_url: str | None) -> bool:
    """True only for explicit Ollama signatures (port 11434 or an ``ollama`` host label).
    ``think`` is Ollama-native; strict hosts (Mistral, Groq) 422 on it, and
    arbitrary localhost may be llama.cpp / vLLM / LM Studio."""
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:  # urlparse raises ValueError on malformed ports ("host:99999"); treat as not-Ollama.
        if parsed.port == 11434:
            return True
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return bool(host) and (host == "ollama.com" or host.endswith(".ollama.com") or "ollama" in host.split("."))


def _mirror_reasoning_effort_into_chat_template_kwargs(
    extra_body: dict[str, Any], effort: str
) -> None:
    """Mirror a resolved ``reasoning_effort`` into ``chat_template_kwargs``.

    Some OpenAI-compatible servers (vLLM/SGLang serving Qwen3-style chat
    templates) read ``reasoning_effort`` only from ``chat_template_kwargs``
    and ignore the top-level field, so the profile emits both. A
    caller-provided ``chat_template_kwargs`` (if present in this dict) is
    deep-merged, and an explicit caller-set ``reasoning_effort`` wins.
    """
    existing = extra_body.get("chat_template_kwargs")
    if not isinstance(existing, dict):
        existing = {}
        extra_body["chat_template_kwargs"] = existing
    existing.setdefault("reasoning_effort", effort)


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, ollama_num_ctx: int | None = None, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if ollama_num_ctx:
            extra_body["options"] = {"num_ctx": ollama_num_ctx}

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → top-level reasoning_effort="none"; extra_body.think
        #     = False only on Ollama URLs (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # Every emitted effort is ALSO mirrored into
        # extra_body.chat_template_kwargs: vLLM/SGLang serving Qwen3-style
        # chat templates read reasoning_effort only from chat_template_kwargs
        # and ignore the top-level field. Endpoints that don't know the field
        # simply ignore it.
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent. The same
        # constraint applies to ``think=False`` on disable — Mistral/Groq
        # reject unknown fields (HTTP 422 extra_forbidden) rather than ignoring
        # them, so that flag stays Ollama-URL-gated.
        if reasoning_config and isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort == "none" or reasoning_config.get("enabled", True) is False:
                # See #14820.
                top_level["reasoning_effort"] = "none"
                if _looks_like_ollama_endpoint(ctx.get("base_url")):
                    extra_body["think"] = False
                _mirror_reasoning_effort_into_chat_template_kwargs(
                    extra_body, "none"
                )
            elif effort:
                # Clamp the internal ladder onto the widest OpenAI-compatible
                # wire vocabulary (shared policy in agent.reasoning_effort) —
                # GLM/ARK, vLLM and SGLang all top out at "max"; forwarding
                # "ultra" verbatim is a guaranteed 400 (#89503).
                top_level["reasoning_effort"] = clamp_effort(
                    effort, OPENAI_COMPAT_WIRE_EFFORTS
                )
                _mirror_reasoning_effort_into_chat_template_kwargs(
                    extra_body, effort
                )

        return extra_body, top_level

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """base_url is user-configured; fetch only if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom", aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # An arbitrary client ceiling can exceed a local server's actual output limit.
    # The endpoint owns its generation default.
)

register_provider(custom)
