"""Unit tests for safe_run.config module.

Covers config loading, defaults, validation edge cases,
allowlist/blocklist matching, and risk threshold comparisons.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from safe_run.config import (
    ConfigError,
    SafeRunConfig,
    _build_config_from_dict,
    _default_config_toml,
    load_config,
    write_default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to use as the config home."""
    return tmp_path


@pytest.fixture()
def minimal_toml_file(tmp_path: Path) -> Path:
    """Write a minimal valid TOML file and return its path."""
    content = """
[llm]
provider = "openai"

[risk]
threshold = "MEDIUM"
"""
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def full_toml_file(tmp_path: Path) -> Path:
    """Write a fully-specified valid TOML file and return its path."""
    content = _default_config_toml()
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SafeRunConfig defaults
# ---------------------------------------------------------------------------


class TestSafeRunConfigDefaults:
    def test_default_llm_provider(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.llm_provider == "openai"

    def test_default_openai_model(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.openai_model == "gpt-4o"

    def test_default_ollama_model(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.ollama_model == "llama3"

    def test_default_ollama_base_url(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.ollama_base_url == "http://localhost:11434/v1"

    def test_default_timeout(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.timeout == 30

    def test_default_risk_threshold(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.risk_threshold == "HIGH"

    def test_default_auto_confirm(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.auto_confirm_below_threshold is True

    def test_default_allowlist_empty(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.allowlist == []

    def test_default_blocklist_empty(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.blocklist == []

    def test_default_max_command_length(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.max_command_length == 2000

    def test_default_show_raw_command(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.show_raw_command is True

    def test_default_log_level(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.log_level == "WARNING"

    def test_default_openai_api_key_none(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.openai_api_key is None

    def test_default_extra_empty(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.extra == {}


# ---------------------------------------------------------------------------
# SafeRunConfig validation
# ---------------------------------------------------------------------------


class TestSafeRunConfigValidation:
    def test_invalid_llm_provider_raises(self) -> None:
        with pytest.raises(ConfigError, match="llm_provider"):
            SafeRunConfig(llm_provider="anthropic")

    def test_invalid_risk_threshold_raises(self) -> None:
        with pytest.raises(ConfigError, match="risk_threshold"):
            SafeRunConfig(risk_threshold="EXTREME")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ConfigError, match="log_level"):
            SafeRunConfig(log_level="VERBOSE")

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ConfigError, match="timeout"):
            SafeRunConfig(timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ConfigError, match="timeout"):
            SafeRunConfig(timeout=-5)

    def test_zero_max_command_length_raises(self) -> None:
        with pytest.raises(ConfigError, match="max_command_length"):
            SafeRunConfig(max_command_length=0)

    def test_allowlist_non_list_raises(self) -> None:
        with pytest.raises(ConfigError, match="allowlist"):
            SafeRunConfig(allowlist="echo")  # type: ignore[arg-type]

    def test_blocklist_non_list_raises(self) -> None:
        with pytest.raises(ConfigError, match="blocklist"):
            SafeRunConfig(blocklist={"rm"})  # type: ignore[arg-type]

    def test_allowlist_non_string_entry_raises(self) -> None:
        with pytest.raises(ConfigError):
            SafeRunConfig(allowlist=[123])  # type: ignore[list-item]

    def test_valid_ollama_provider(self) -> None:
        cfg = SafeRunConfig(llm_provider="ollama")
        assert cfg.llm_provider == "ollama"

    def test_all_valid_risk_levels(self) -> None:
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            cfg = SafeRunConfig(risk_threshold=level)
            assert cfg.risk_threshold == level


# ---------------------------------------------------------------------------
# effective_openai_api_key
# ---------------------------------------------------------------------------


class TestEffectiveOpenAIApiKey:
    def test_returns_config_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig(openai_api_key="sk-test")
        assert cfg.effective_openai_api_key == "sk-test"

    def test_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cfg = SafeRunConfig(openai_api_key=None)
        assert cfg.effective_openai_api_key == "sk-from-env"

    def test_returns_none_when_neither_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig(openai_api_key=None)
        assert cfg.effective_openai_api_key is None

    def test_config_value_takes_priority_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = SafeRunConfig(openai_api_key="sk-config")
        assert cfg.effective_openai_api_key == "sk-config"


# ---------------------------------------------------------------------------
# is_above_threshold
# ---------------------------------------------------------------------------


class TestIsAboveThreshold:
    def test_same_level_as_threshold(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("HIGH") is True

    def test_level_above_threshold(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("CRITICAL") is True

    def test_level_below_threshold(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("LOW") is False

    def test_level_just_below_threshold(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("MEDIUM") is False

    def test_low_threshold_all_pass(self) -> None:
        cfg = SafeRunConfig(risk_threshold="LOW")
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert cfg.is_above_threshold(level) is True

    def test_critical_threshold_only_critical_passes(self) -> None:
        cfg = SafeRunConfig(risk_threshold="CRITICAL")
        assert cfg.is_above_threshold("CRITICAL") is True
        for level in ("LOW", "MEDIUM", "HIGH"):
            assert cfg.is_above_threshold(level) is False

    def test_unknown_level_raises(self) -> None:
        cfg = SafeRunConfig()
        with pytest.raises(ConfigError, match="Unknown risk level"):
            cfg.is_above_threshold("EXTREME")


# ---------------------------------------------------------------------------
# is_allowlisted / is_blocklisted
# ---------------------------------------------------------------------------


class TestAllowAndBlocklist:
    def test_allowlist_match(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo", "ls"])
        assert cfg.is_allowlisted("echo hello") is True

    def test_allowlist_no_match(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo", "ls"])
        assert cfg.is_allowlisted("rm -rf /") is False

    def test_allowlist_exact_match(self) -> None:
        cfg = SafeRunConfig(allowlist=["ls"])
        assert cfg.is_allowlisted("ls") is True

    def test_allowlist_strips_leading_whitespace(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo"])
        assert cfg.is_allowlisted("  echo hello") is True

    def test_blocklist_match(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm -rf"])
        assert cfg.is_blocklisted("rm -rf /") is True

    def test_blocklist_no_match(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm -rf"])
        assert cfg.is_blocklisted("ls /") is False

    def test_empty_allowlist_never_matches(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.is_allowlisted("anything") is False

    def test_empty_blocklist_never_matches(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.is_blocklisted("anything") is False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_returns_defaults_when_no_file_exists(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        cfg = load_config(missing)
        assert isinstance(cfg, SafeRunConfig)
        assert cfg.llm_provider == "openai"

    def test_loads_minimal_file(self, minimal_toml_file: Path) -> None:
        cfg = load_config(minimal_toml_file)
        assert cfg.risk_threshold == "MEDIUM"

    def test_loads_full_default_toml(self, full_toml_file: Path) -> None:
        cfg = load_config(full_toml_file)
        assert cfg.llm_provider == "openai"
        assert cfg.risk_threshold == "HIGH"
        assert cfg.timeout == 30

    def test_invalid_toml_raises_config_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.toml"
        bad_file.write_text("[[[not valid toml", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad_file)

    def test_env_var_overrides_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "custom.toml"
        cfg_file.write_text(
            "[risk]\nthreshold = \"CRITICAL\"\n", encoding="utf-8"
        )
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(cfg_file))
        cfg = load_config()
        assert cfg.risk_threshold == "CRITICAL"

    def test_explicit_path_overrides_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / "env.toml"
        env_file.write_text(
            "[risk]\nthreshold = \"CRITICAL\"\n", encoding="utf-8"
        )
        explicit_file = tmp_path / "explicit.toml"
        explicit_file.write_text(
            "[risk]\nthreshold = \"LOW\"\n", encoding="utf-8"
        )
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(env_file))
        cfg = load_config(explicit_file)
        assert cfg.risk_threshold == "LOW"

    def test_unreadable_file_raises_config_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[llm]\n", encoding="utf-8")
        cfg_file.chmod(0o000)
        try:
            with pytest.raises(ConfigError, match="Cannot read"):
                load_config(cfg_file)
        finally:
            cfg_file.chmod(0o644)


# ---------------------------------------------------------------------------
# _build_config_from_dict
# ---------------------------------------------------------------------------


class TestBuildConfigFromDict:
    def test_empty_dict_uses_all_defaults(self) -> None:
        cfg = _build_config_from_dict({})
        assert cfg.llm_provider == "openai"
        assert cfg.risk_threshold == "HIGH"

    def test_llm_section_parsed(self) -> None:
        raw = {"llm": {"provider": "ollama", "ollama_model": "mistral"}}
        cfg = _build_config_from_dict(raw)
        assert cfg.llm_provider == "ollama"
        assert cfg.ollama_model == "mistral"

    def test_risk_section_parsed(self) -> None:
        raw = {
            "risk": {
                "threshold": "MEDIUM",
                "allowlist": ["echo"],
                "auto_confirm_below_threshold": False,
            }
        }
        cfg = _build_config_from_dict(raw)
        assert cfg.risk_threshold == "MEDIUM"
        assert cfg.allowlist == ["echo"]
        assert cfg.auto_confirm_below_threshold is False

    def test_display_section_parsed(self) -> None:
        raw = {"display": {"show_raw_command": False}}
        cfg = _build_config_from_dict(raw)
        assert cfg.show_raw_command is False

    def test_logging_section_parsed(self) -> None:
        raw = {"logging": {"level": "DEBUG"}}
        cfg = _build_config_from_dict(raw)
        assert cfg.log_level == "DEBUG"

    def test_commands_section_parsed(self) -> None:
        raw = {"commands": {"max_command_length": 500}}
        cfg = _build_config_from_dict(raw)
        assert cfg.max_command_length == 500

    def test_unknown_top_level_keys_go_to_extra(self) -> None:
        raw = {"unknown_future_section": {"key": "value"}}
        cfg = _build_config_from_dict(raw)
        assert "unknown_future_section" in cfg.extra

    def test_invalid_section_type_raises(self) -> None:
        raw = {"llm": "not a table"}
        with pytest.raises(ConfigError, match="llm"):
            _build_config_from_dict(raw)  # type: ignore[arg-type]

    def test_wrong_type_for_timeout_raises(self) -> None:
        raw = {"llm": {"timeout": "thirty"}}
        with pytest.raises(ConfigError, match="timeout"):
            _build_config_from_dict(raw)

    def test_wrong_type_for_threshold_raises(self) -> None:
        raw = {"risk": {"threshold": 42}}
        with pytest.raises(ConfigError, match="threshold"):
            _build_config_from_dict(raw)

    def test_allowlist_with_non_string_raises(self) -> None:
        raw = {"risk": {"allowlist": [1, 2, 3]}}
        with pytest.raises(ConfigError):
            _build_config_from_dict(raw)

    def test_openai_api_key_empty_string_becomes_none(self) -> None:
        raw = {"llm": {"openai_api_key": ""}}
        cfg = _build_config_from_dict(raw)
        assert cfg.openai_api_key is None


# ---------------------------------------------------------------------------
# write_default_config
# ---------------------------------------------------------------------------


class TestWriteDefaultConfig:
    def test_creates_file_at_given_path(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "config.toml"
        result = write_default_config(target)
        assert result == target
        assert target.exists()

    def test_created_file_is_valid_toml_and_loadable(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        cfg = load_config(target)
        assert isinstance(cfg, SafeRunConfig)

    def test_does_not_overwrite_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("[risk]\nthreshold = \"LOW\"\n", encoding="utf-8")
        write_default_config(target)
        content = target.read_text(encoding="utf-8")
        # Original content should be preserved
        assert 'threshold = "LOW"' in content

    def test_returns_existing_path_without_modification(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("# existing", encoding="utf-8")
        result = write_default_config(target)
        assert result == target
        assert target.read_text(encoding="utf-8") == "# existing"

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "config.toml"
        write_default_config(target)
        assert target.exists()


# ---------------------------------------------------------------------------
# _default_config_toml
# ---------------------------------------------------------------------------


class TestDefaultConfigToml:
    def test_contains_all_sections(self) -> None:
        toml_str = _default_config_toml()
        for section in ("[llm]", "[risk]", "[display]", "[logging]", "[commands]"):
            assert section in toml_str

    def test_is_parseable_toml(self) -> None:
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-reuse-of-import]

        toml_str = _default_config_toml()
        data = tomllib.loads(toml_str)
        assert isinstance(data, dict)

    def test_contains_provider_key(self) -> None:
        toml_str = _default_config_toml()
        assert "provider" in toml_str

    def test_contains_threshold_key(self) -> None:
        toml_str = _default_config_toml()
        assert "threshold" in toml_str
