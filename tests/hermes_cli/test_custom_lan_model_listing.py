"""Custom provider on a LAN address: the picker's model list must stay live.

A llama.cpp/Ollama router on the local network loads/unloads models at operator
timescales; the ``provider_models_cache.json`` row for the ``custom`` slug (1h TTL
+ 7-day stale-while-revalidate) must never serve such a server's lineup as current.
With ``model.base_url`` pointing at a LAN host, every ``cached_provider_model_ids()``
read re-fetches live instead of hitting the TTL/SWR cache. Remote custom endpoints
keep the existing cache contract (fresh rows still short-circuit).
"""

import os
from pathlib import Path

import yaml


def _write_config(base_url: str) -> None:
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": "custom", "base_url": base_url, "api_key": "k"}},
                       allow_unicode=True),
        encoding="utf-8",
    )


class TestCustomLanModelListingLiveOnly:
    def test_lan_custom_slug_always_live_even_within_ttl(self, monkeypatch):
        """A row written seconds ago must not be served for a LAN endpoint: the second
        read within the 1h TTL still re-fetches."""
        import hermes_cli.models as mod

        _write_config("http://chibigamer.lan:1235/v1")
        assert mod._custom_base_url_is_local() is True  # gate engages for a *.lan base_url

        calls = []

        def fake_pmi(provider, *, force_refresh=False):
            calls.append(provider)
            return ["live-a"] if len(calls) == 1 else ["live-a", "live-b"]

        monkeypatch.setattr(mod, "provider_model_ids", fake_pmi)
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda p: "fp")

        first = mod.cached_provider_model_ids("custom")
        second = mod.cached_provider_model_ids("custom")

        assert first == ["live-a"]
        assert second == ["live-a", "live-b"]
        assert len(calls) == 2, "a fresh cached row must NOT short-circuit a LAN read"

    def test_remote_custom_slug_still_served_from_fresh_row(self, monkeypatch):
        """The live-only gate is scoped to LAN base_urls: a remote custom endpoint keeps
        the TTL contract (second read within the TTL comes from the cache)."""
        import hermes_cli.models as mod

        _write_config("https://gw.example.com/v1")
        assert mod._custom_base_url_is_local() is False

        calls = []

        def fake_pmi(provider, *, force_refresh=False):
            calls.append(provider)
            return ["remote-model"]

        monkeypatch.setattr(mod, "provider_model_ids", fake_pmi)
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda p: "fp")

        assert mod.cached_provider_model_ids("custom") == ["remote-model"]
        assert mod.cached_provider_model_ids("custom") == ["remote-model"]
        assert len(calls) == 1, "remote custom endpoints keep the fresh-row short-circuit"

    def test_lan_offline_read_falls_back_to_last_snapshot(self, monkeypatch):
        """LAN rows are still persisted, so a stopped box answers with its last known
        catalog instead of blanking the picker (stale beats nothing)."""
        import hermes_cli.models as mod

        _write_config("http://chibigamer.lan:1235/v1")

        def down(provider, *, force_refresh=False):
            return None

        monkeypatch.setattr(mod, "_credential_fingerprint", lambda p: "fp")

        monkeypatch.setattr(mod, "provider_model_ids", lambda *a, **k: ["last-seen"])
        assert mod.cached_provider_model_ids("custom") == ["last-seen"]

        monkeypatch.setattr(mod, "provider_model_ids", down)  # box is down
        assert mod.cached_provider_model_ids("custom") == ["last-seen"]