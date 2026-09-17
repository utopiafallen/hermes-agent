"""Regression tests: Qwen3.6 / Qwen3.8 / Ornith reasoning_content echo for
preserve_thinking.

Qwen3.6, Qwen3.8, and Ornith expose a ``preserve_thinking`` feature whose chat
template renders `` tags from the prior turns' reasoning. That requires the
replayed assistant messages to carry ``reasoning_content`` — the internal
'reasoning' field is promoted to 'reasoning_content' at replay time, and
tool-call turns that were persisted without either get a single-space pad.

Detection is model-name-based: any model containing "qwen3.6", "qwen3.8", or
"ornith" (e.g. "qwen3.6-plus", "qwen3.6-27b", "qwen3.8-max",
"Ornith-1.5-35B-A3B") opts into the pad. Siblings (qwen3.5-plus, qwen3.7-max,
qwen3-coder) do not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


class TestNeedsQwenThinkingPad:
    """_needs_qwen_thinking_pad() recognises qwen3.6, qwen3.8 and ornith model names."""

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("custom", "qwen3.8-max", ""),
            ("custom", "qwen3.8", ""),
            ("custom", "qwen3.6-plus", ""),
            ("custom", "qwen3.6-27b", ""),
            ("custom", "qwen3.6-35b-a3b", ""),
            ("openrouter", "qwen/qwen3.8-max", "https://openrouter.ai/api/v1"),
            ("nous", "qwen3.8-max", "https://portal.nousresearch.com/v1"),
            ("custom", "Ornith-1.5-35B-A3B", ""),
            ("custom", "unsloth/Ornith-1.5-35B-A3B:Q6_K", ""),
        ],
    )
    def test_qwen36_qwen38_models_match(self, provider, model, base_url) -> None:
        agent = _make_agent(provider=provider, model=model, base_url=base_url)
        assert agent._needs_qwen_thinking_pad() is True

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("custom", "qwen3.5-plus", ""),
            ("custom", "qwen3.7-max", ""),
            ("custom", "qwen3-coder", ""),
            ("custom", "qwen-235b", ""),
            ("openai", "gpt-5", "https://api.openai.com/v1"),
        ],
    )
    def test_other_models_do_not_match(self, provider, model, base_url) -> None:
        agent = _make_agent(provider=provider, model=model, base_url=base_url)
        assert agent._needs_qwen_thinking_pad() is False


class TestNeedsThinkingReasoningPadQwen:
    """_needs_thinking_reasoning_pad() is True for Qwen3.6/Qwen3.8 and False
    for sibling models, and the per-instance cache follows switch_model-style
    mutation of (provider, model, base_url)."""

    def test_qwen38_max_is_padded(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.8-max")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_qwen36_plus_is_padded(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.6-plus")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_ornith_is_padded(self) -> None:
        agent = _make_agent(provider="custom", model="unsloth/Ornith-1.5-35B-A3B:Q6_K")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_qwen37_max_is_not_padded(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.7-max")
        assert agent._needs_thinking_reasoning_pad() is False

    def test_cache_invalidates_on_model_switch(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.8-max")
        assert agent._needs_thinking_reasoning_pad() is True
        # Simulate switch_model(): the detection key changes, so the cached
        # True from the Qwen3.8 model must not leak into the next model.
        agent.model = "qwen3.7-max"
        assert agent._needs_thinking_reasoning_pad() is False
        agent.model = "qwen3.6-plus"
        assert agent._needs_thinking_reasoning_pad() is True


class TestCopyReasoningContentForApiQwen:
    """_copy_reasoning_content_for_api() promotes 'reasoning' to
    'reasoning_content' on Qwen3.6/Qwen3.8 replay, and strips the field on
    sibling models (strict providers reject the key outright)."""

    def test_qwen38_promotes_reasoning_field(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.8-max")
        source = {
            "role": "assistant",
            "content": "done",
            "reasoning": "chain of thought",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == "chain of thought"

    def test_ornith_promotes_reasoning_field(self) -> None:
        agent = _make_agent(provider="custom", model="unsloth/Ornith-1.5-35B-A3B:Q6_K")
        source = {
            "role": "assistant",
            "content": "done",
            "reasoning": "chain of thought",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == "chain of thought"

    def test_qwen36_preserves_existing_reasoning_content(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.6-plus")
        source = {
            "role": "assistant",
            "content": "done",
            "reasoning_content": "prior turn reasoning",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == "prior turn reasoning"

    def test_qwen38_tool_call_turn_gets_pad(self) -> None:
        """Tool-call turns persisted without any reasoning content get a
        single-space pad so preserve_thinking replay does not 400."""
        agent = _make_agent(provider="custom", model="qwen3.8-max")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_qwen37_sibling_strips_reasoning_content(self) -> None:
        """qwen3.7-max is outside the preserve_thinking family — the field
        must not leak into its requests (strict APIs reject the key)."""
        agent = _make_agent(provider="custom", model="qwen3.7-max")
        source = {
            "role": "assistant",
            "content": "hi",
            "reasoning_content": "stale pad",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_user_messages_are_untouched(self) -> None:
        agent = _make_agent(provider="custom", model="qwen3.8-max")
        source = {"role": "user", "content": "hi", "reasoning": "x"}
        api_msg: dict = {"role": "user", "content": "hi"}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg
