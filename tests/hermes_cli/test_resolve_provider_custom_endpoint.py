"""A configured custom (OpenAI-compatible) endpoint is explicit provider intent.

Regression for #108383: ``resolve_provider("auto")`` recognised only registry providers from
``model.provider``, so the boot inventory (``free_tier_bootstrap``) read a llama.cpp / vLLM /
ollama install as "nothing configured" and ``setup.status`` reported ``provider_configured:
False`` — the dashboard's Ink chat parked every new session on "Setup Required" while
``hermes chat`` (which resolves the runtime directly) worked against the same config.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_GUEST_ONBOARDING", raising=False)
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL",
                "OPENROUTER_BASE_URL", "HERMES_INFERENCE_PROVIDER", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
    from hermes_cli import free_tier_bootstrap as fb
    fb.reset_for_tests()
    return home


@pytest.mark.parametrize(
    "model_block",
    [
        pytest.param(
            "model:\n  default: nvidia/Nemotron\n  provider: custom\n"
            "  base_url: http://127.0.0.1:8000/v1\n  api_key: dummy\n",
            id="provider-custom",
        ),
        pytest.param(
            "model:\n  default: qwen3\n  provider: vllm\n  base_url: http://127.0.0.1:8000/v1\n",
            id="local-server-alias",
        ),
        pytest.param(
            "model:\n  default: qwen3\n  base_url: http://localhost:8080/v1\n",
            id="loopback-base-url-only",
        ),
        pytest.param(
            "model:\n  default: openrouter/auto\n  provider: openrouter\n",
            id="provider-openrouter",
        ),
        pytest.param(
            "model:\n  default: openrouter/auto\n  provider: openrouter\n"
            "  base_url: https://openrouter-mirror.example.com/api/v1\n",
            id="provider-openrouter-mirror",
        ),
    ],
)
def test_configured_custom_endpoint_resolves_as_a_provider(isolated_home, model_block):
    (isolated_home / "config.yaml").write_text(model_block, encoding="utf-8")
    from hermes_cli.auth import resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    assert resolve_provider("auto") in ("custom", "openrouter")
    record = run_bootstrap(announce=False)
    assert record.provider_configured is True
    assert record.other_providers is True


def test_configured_openrouter_pin_resolves_as_a_provider(isolated_home):
    """``model.provider: openrouter`` is explicit intent like a registry pin, not "nothing
    configured".

    Regression for #109397: ``_config_model_provider()`` recognised ``custom`` but not
    ``openrouter`` (both are intentionally absent from PROVIDER_REGISTRY), so the boot inventory
    parked dashboard sessions on Setup Required for openrouter-pinned installs.
    """
    (isolated_home / "config.yaml").write_text(
        "model:\n  default: openrouter/auto\n  provider: openrouter\n", encoding="utf-8",
    )
    from hermes_cli.auth import resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    assert resolve_provider("auto") == "openrouter"
    record = run_bootstrap(announce=False)
    assert record.provider_configured is True
    assert record.other_providers is True
    assert record.inference_provider == "openrouter"


def test_named_provider_pin_resolves_as_a_provider(isolated_home):
    """A bare ``model.provider`` naming a ``providers:`` entry is explicit intent too.

    Sibling of the openrouter rung above: ``has_named_custom_provider()`` already routes this
    config at runtime (``hermes chat`` works against ``providers.CPA``), but the boot inventory
    discarded the bare name, so ``setup.status`` reported ``provider_configured: False`` and the
    dashboard parked sessions on Setup Required for a working endpoint.
    """
    (isolated_home / "config.yaml").write_text(
        "model:\n  default: test-model\n  provider: CPA\n\n"
        "providers:\n  CPA:\n    api: http://127.0.0.1:8317/v1\n    default_model: test-model\n",
        encoding="utf-8",
    )
    from hermes_cli.auth import resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    assert resolve_provider("auto") == "custom"
    record = run_bootstrap(announce=False)
    assert record.provider_configured is True
    assert record.other_providers is True


def test_stale_remote_base_url_without_a_custom_pin_is_not_a_provider(isolated_home):
    """The URL rung follows the runtime's own trust rule: a non-loopback ``base_url`` left behind
    under a bare (unpinned) provider is not custom intent (#14676), so a blank machine still reads
    as unconfigured. (A ``provider: openrouter`` pin is excluded from this guard — a non-openrouter
    ``base_url`` under it is a deliberate mirror/proxy, #10622/#109397.)"""
    (isolated_home / "config.yaml").write_text(
        "model:\n  default: some/model\n  base_url: https://api.z.ai/v1\n",
        encoding="utf-8",
    )
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    with pytest.raises(AuthError):
        resolve_provider("auto")
    assert run_bootstrap(announce=False).provider_configured is False


def test_auto_provider_with_loopback_base_url_resolves_without_recursing(isolated_home, monkeypatch):
    """A fresh setup keeps ``provider: auto`` until the picker stores its choice (#110926)."""
    (isolated_home / "config.yaml").write_text(
        "model:\n  provider: auto\n  base_url: http://127.0.0.1:8000/v1\n",
        encoding="utf-8",
    )
    from hermes_cli import runtime_provider
    from hermes_cli.auth import resolve_provider

    def unexpected_provider_resolution(_name):
        raise AssertionError("the bare custom trust check must not resolve model.provider=auto")

    monkeypatch.setattr(runtime_provider, "_resolves_to_custom", unexpected_provider_resolution)

    assert resolve_provider("auto") == "custom"
