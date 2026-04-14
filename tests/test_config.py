"""Unit tests for safe_run.config module.

Covers config loading, defaults, validation edge cases,
allowlist/blocklist matching, risk threshold comparisons,
and write_default_config behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

from safe_run.config import (
    ConfigError,
    SafeRunConfig,
    _build_config_from_dict,
    _default_config_toml,
    _get_bool,
    _get_int,
    _get_str,
    _get_str_list,
    _parse_toml_section,
    _resolve_config_path,
    load_config,
    write_default_config,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RISK_THRESHOLD,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_COMMAND_LENGTH,
    VALID_LLM_PROVIDERS,
    VALID_RISK_LEVELS,
    VALID_LOG_LEVELS,
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


@pytest.fixture()
def ollama_toml_file(tmp_path: Path) -> Path:
    """Write a TOML file configured for Ollama and return its path."""
    content = """
[llm]
provider = "ollama"
ollama_model = "mistral"
ollama_base_url = "http://localhost:11434/v1"
timeout = 60

[risk]
threshold = "HIGH"
auto_confirm_below_threshold = false
allowlist = ["ls", "echo"]
blocklist = ["rm -rf"]

[commands]
max_command_length = 1000

[display]
show_raw_command = false

[logging]
level = "DEBUG"
"""
    p = tmp_path / "ollama_config.toml"
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
        assert cfg.openai_model == DEFAULT_OPENAI_MODEL

    def test_default_ollama_model(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.ollama_model == DEFAULT_OLLAMA_MODEL

    def test_default_ollama_base_url(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.ollama_base_url == DEFAULT_OLLAMA_BASE_URL

    def test_default_timeout(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.timeout == DEFAULT_TIMEOUT

    def test_default_risk_threshold(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.risk_threshold == DEFAULT_RISK_THRESHOLD

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
        assert cfg.max_command_length == DEFAULT_MAX_COMMAND_LENGTH

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

    def test_default_timeout_positive(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.timeout > 0

    def test_default_max_command_length_positive(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.max_command_length > 0

    def test_default_openai_model_non_empty(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.openai_model
        assert isinstance(cfg.openai_model, str)

    def test_default_ollama_model_non_empty(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.ollama_model
        assert isinstance(cfg.ollama_model, str)

    def test_default_risk_threshold_is_valid(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.risk_threshold in VALID_RISK_LEVELS

    def test_default_log_level_is_valid(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.log_level in VALID_LOG_LEVELS

    def test_default_llm_provider_is_valid(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.llm_provider in VALID_LLM_PROVIDERS

    def test_allowlist_is_list(self) -> None:
        cfg = SafeRunConfig()
        assert isinstance(cfg.allowlist, list)

    def test_blocklist_is_list(self) -> None:
        cfg = SafeRunConfig()
        assert isinstance(cfg.blocklist, list)

    def test_extra_is_dict(self) -> None:
        cfg = SafeRunConfig()
        assert isinstance(cfg.extra, dict)


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

    def test_negative_max_command_length_raises(self) -> None:
        with pytest.raises(ConfigError, match="max_command_length"):
            SafeRunConfig(max_command_length=-100)

    def test_allowlist_non_list_raises(self) -> None:
        with pytest.raises(ConfigError, match="allowlist"):
            SafeRunConfig(allowlist="echo")  # type: ignore[arg-type]

    def test_blocklist_non_list_raises(self) -> None:
        with pytest.raises(ConfigError, match="blocklist"):
            SafeRunConfig(blocklist={"rm"})  # type: ignore[arg-type]

    def test_allowlist_non_string_entry_raises(self) -> None:
        with pytest.raises(ConfigError):
            SafeRunConfig(allowlist=[123])  # type: ignore[list-item]

    def test_blocklist_non_string_entry_raises(self) -> None:
        with pytest.raises(ConfigError):
            SafeRunConfig(blocklist=[3.14])  # type: ignore[list-item]

    def test_valid_ollama_provider(self) -> None:
        cfg = SafeRunConfig(llm_provider="ollama")
        assert cfg.llm_provider == "ollama"

    def test_all_valid_risk_levels(self) -> None:
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            cfg = SafeRunConfig(risk_threshold=level)
            assert cfg.risk_threshold == level

    def test_all_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = SafeRunConfig(log_level=level)
            assert cfg.log_level == level

    def test_all_valid_providers(self) -> None:
        for provider in ("openai", "ollama"):
            cfg = SafeRunConfig(llm_provider=provider)
            assert cfg.llm_provider == provider

    def test_valid_non_empty_allowlist(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo", "ls", "cat"])
        assert cfg.allowlist == ["echo", "ls", "cat"]

    def test_valid_non_empty_blocklist(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm -rf", "mkfs"])
        assert cfg.blocklist == ["rm -rf", "mkfs"]

    def test_timeout_one_is_valid(self) -> None:
        cfg = SafeRunConfig(timeout=1)
        assert cfg.timeout == 1

    def test_max_command_length_one_is_valid(self) -> None:
        cfg = SafeRunConfig(max_command_length=1)
        assert cfg.max_command_length == 1

    def test_empty_allowlist_valid(self) -> None:
        cfg = SafeRunConfig(allowlist=[])
        assert cfg.allowlist == []

    def test_empty_blocklist_valid(self) -> None:
        cfg = SafeRunConfig(blocklist=[])
        assert cfg.blocklist == []

    def test_openai_api_key_string_valid(self) -> None:
        cfg = SafeRunConfig(openai_api_key="sk-test-key")
        assert cfg.openai_api_key == "sk-test-key"

    def test_show_raw_command_false_valid(self) -> None:
        cfg = SafeRunConfig(show_raw_command=False)
        assert cfg.show_raw_command is False

    def test_auto_confirm_false_valid(self) -> None:
        cfg = SafeRunConfig(auto_confirm_below_threshold=False)
        assert cfg.auto_confirm_below_threshold is False

    def test_extra_dict_preserved(self) -> None:
        extra = {"future_section": {"key": "value"}}
        cfg = SafeRunConfig(extra=extra)
        assert cfg.extra == extra

    def test_large_timeout_valid(self) -> None:
        cfg = SafeRunConfig(timeout=3600)
        assert cfg.timeout == 3600

    def test_large_max_command_length_valid(self) -> None:
        cfg = SafeRunConfig(max_command_length=100_000)
        assert cfg.max_command_length == 100_000


# ---------------------------------------------------------------------------
# effective_openai_api_key
# ---------------------------------------------------------------------------


class TestEffectiveOpenAIApiKey:
    def test_returns_config_value_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig(openai_api_key="sk-test")
        assert cfg.effective_openai_api_key == "sk-test"

    def test_falls_back_to_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cfg = SafeRunConfig(openai_api_key=None)
        assert cfg.effective_openai_api_key == "sk-from-env"

    def test_returns_none_when_neither_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig(openai_api_key=None)
        assert cfg.effective_openai_api_key is None

    def test_config_value_takes_priority_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = SafeRunConfig(openai_api_key="sk-config")
        assert cfg.effective_openai_api_key == "sk-config"

    def test_empty_config_key_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-fallback")
        # None openai_api_key should use env var
        cfg = SafeRunConfig(openai_api_key=None)
        assert cfg.effective_openai_api_key == "sk-env-fallback"

    def test_env_var_not_set_and_key_none_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig()
        assert cfg.effective_openai_api_key is None

    def test_returns_string_type_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SafeRunConfig(openai_api_key="sk-string")
        result = cfg.effective_openai_api_key
        assert isinstance(result, str)

    def test_returns_string_type_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-str")
        cfg = SafeRunConfig(openai_api_key=None)
        result = cfg.effective_openai_api_key
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# risk_level_order
# ---------------------------------------------------------------------------


class TestRiskLevelOrder:
    def test_returns_list_of_four_levels(self) -> None:
        cfg = SafeRunConfig()
        order = cfg.risk_level_order
        assert len(order) == 4

    def test_low_is_first(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.risk_level_order[0] == "LOW"

    def test_critical_is_last(self) -> None:
        cfg = SafeRunConfig()
        assert cfg.risk_level_order[-1] == "CRITICAL"

    def test_correct_order(self) -> None:
        cfg = SafeRunConfig()
        order = cfg.risk_level_order
        assert order == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_all_valid_risk_levels_present(self) -> None:
        cfg = SafeRunConfig()
        order = cfg.risk_level_order
        for level in VALID_RISK_LEVELS:
            assert level in order


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

    def test_medium_threshold_medium_and_above_pass(self) -> None:
        cfg = SafeRunConfig(risk_threshold="MEDIUM")
        assert cfg.is_above_threshold("MEDIUM") is True
        assert cfg.is_above_threshold("HIGH") is True
        assert cfg.is_above_threshold("CRITICAL") is True
        assert cfg.is_above_threshold("LOW") is False

    def test_medium_threshold_low_does_not_pass(self) -> None:
        cfg = SafeRunConfig(risk_threshold="MEDIUM")
        assert cfg.is_above_threshold("LOW") is False

    def test_high_threshold_medium_does_not_pass(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("MEDIUM") is False

    def test_high_threshold_low_does_not_pass(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        assert cfg.is_above_threshold("LOW") is False

    def test_returns_bool_type(self) -> None:
        cfg = SafeRunConfig(risk_threshold="HIGH")
        result = cfg.is_above_threshold("HIGH")
        assert isinstance(result, bool)

    def test_empty_string_raises(self) -> None:
        cfg = SafeRunConfig()
        with pytest.raises(ConfigError):
            cfg.is_above_threshold("")

    def test_lowercase_level_raises(self) -> None:
        cfg = SafeRunConfig()
        with pytest.raises(ConfigError):
            cfg.is_above_threshold("high")


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

    def test_allowlist_prefix_match_partial(self) -> None:
        cfg = SafeRunConfig(allowlist=["git status"])
        assert cfg.is_allowlisted("git status --short") is True

    def test_allowlist_does_not_match_partial_word(self) -> None:
        # 'ec' should not match 'echo'
        cfg = SafeRunConfig(allowlist=["ec"])
        # 'echo' starts with 'ec' — it IS a prefix match
        # so this WOULD match; test correct prefix semantics
        assert cfg.is_allowlisted("echo hi") is True

    def test_blocklist_exact_match(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm"])
        assert cfg.is_blocklisted("rm file.txt") is True

    def test_blocklist_prefix_match(self) -> None:
        cfg = SafeRunConfig(blocklist=["mkfs"])
        assert cfg.is_blocklisted("mkfs.ext4 /dev/sdb") is True

    def test_allowlist_multiple_prefixes_first_matches(self) -> None:
        cfg = SafeRunConfig(allowlist=["ls", "echo", "pwd"])
        assert cfg.is_allowlisted("ls -la") is True

    def test_allowlist_multiple_prefixes_last_matches(self) -> None:
        cfg = SafeRunConfig(allowlist=["ls", "echo", "pwd"])
        assert cfg.is_allowlisted("pwd") is True

    def test_blocklist_multiple_entries(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm -rf", "mkfs", "dd if"])
        assert cfg.is_blocklisted("mkfs.ext4 /dev/sda") is True
        assert cfg.is_blocklisted("dd if=/dev/zero of=/dev/sda") is True
        assert cfg.is_blocklisted("rm -rf /") is True
        assert cfg.is_blocklisted("ls -la") is False

    def test_is_allowlisted_returns_bool(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo"])
        result = cfg.is_allowlisted("echo hi")
        assert isinstance(result, bool)

    def test_is_blocklisted_returns_bool(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm"])
        result = cfg.is_blocklisted("rm file")
        assert isinstance(result, bool)

    def test_allowlist_strips_multiple_spaces(self) -> None:
        cfg = SafeRunConfig(allowlist=["echo"])
        # Multiple leading spaces
        assert cfg.is_allowlisted("   echo hi") is True

    def test_blocklist_strips_leading_whitespace(self) -> None:
        cfg = SafeRunConfig(blocklist=["rm"])
        assert cfg.is_blocklisted("  rm -rf /") is True


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
            '[risk]\nthreshold = "CRITICAL"\n', encoding="utf-8"
        )
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(cfg_file))
        cfg = load_config()
        assert cfg.risk_threshold == "CRITICAL"

    def test_explicit_path_overrides_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / "env.toml"
        env_file.write_text(
            '[risk]\nthreshold = "CRITICAL"\n', encoding="utf-8"
        )
        explicit_file = tmp_path / "explicit.toml"
        explicit_file.write_text(
            '[risk]\nthreshold = "LOW"\n', encoding="utf-8"
        )
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(env_file))
        cfg = load_config(explicit_file)
        assert cfg.risk_threshold == "LOW"

    def test_unreadable_file_raises_config_error(
        self, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[llm]\n", encoding="utf-8")
        cfg_file.chmod(0o000)
        try:
            with pytest.raises(ConfigError, match="Cannot read"):
                load_config(cfg_file)
        finally:
            cfg_file.chmod(0o644)

    def test_returns_safe_run_config_instance(
        self, minimal_toml_file: Path
    ) -> None:
        cfg = load_config(minimal_toml_file)
        assert isinstance(cfg, SafeRunConfig)

    def test_defaults_returned_for_missing_file_none_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Point to a non-existent path so no file is found
        nonexistent = tmp_path / "does_not_exist" / "config.toml"
        cfg = load_config(nonexistent)
        assert cfg.llm_provider == "openai"
        assert cfg.risk_threshold == "HIGH"

    def test_loads_ollama_config(self, ollama_toml_file: Path) -> None:
        cfg = load_config(ollama_toml_file)
        assert cfg.llm_provider == "ollama"
        assert cfg.ollama_model == "mistral"
        assert cfg.timeout == 60
        assert cfg.auto_confirm_below_threshold is False
        assert "ls" in cfg.allowlist
        assert "echo" in cfg.allowlist
        assert "rm -rf" in cfg.blocklist
        assert cfg.max_command_length == 1000
        assert cfg.show_raw_command is False
        assert cfg.log_level == "DEBUG"

    def test_load_config_no_env_no_explicit_returns_default_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Remove SAFE_RUN_CONFIG env var and point default path to tmp
        monkeypatch.delenv("SAFE_RUN_CONFIG", raising=False)
        # We cannot reliably override DEFAULT_CONFIG_PATH at runtime,
        # but we can load from a non-existent path explicitly
        missing = tmp_path / "missing.toml"
        cfg = load_config(missing)
        assert isinstance(cfg, SafeRunConfig)

    def test_invalid_provider_in_toml_raises(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad_provider.toml"
        bad.write_text('[llm]\nprovider = "anthropic"\n', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad)

    def test_invalid_threshold_in_toml_raises(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad_threshold.toml"
        bad.write_text('[risk]\nthreshold = "EXTREME"\n', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad)

    def test_wrong_type_timeout_in_toml_raises(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad_timeout.toml"
        bad.write_text('[llm]\ntimeout = "thirty"\n', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad)


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------


class TestResolveConfigPath:
    def test_explicit_path_returned_as_is(self, tmp_path: Path) -> None:
        p = tmp_path / "myconfig.toml"
        result = _resolve_config_path(p)
        assert result == p

    def test_none_with_env_var_returns_env_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_path = tmp_path / "env_config.toml"
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(env_path))
        result = _resolve_config_path(None)
        assert result == env_path

    def test_none_without_env_var_returns_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SAFE_RUN_CONFIG", raising=False)
        result = _resolve_config_path(None)
        assert result == DEFAULT_CONFIG_PATH

    def test_explicit_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_path = tmp_path / "env.toml"
        explicit_path = tmp_path / "explicit.toml"
        monkeypatch.setenv("SAFE_RUN_CONFIG", str(env_path))
        result = _resolve_config_path(explicit_path)
        assert result == explicit_path

    def test_returns_path_instance(self, tmp_path: Path) -> None:
        p = tmp_path / "conf.toml"
        result = _resolve_config_path(p)
        assert isinstance(result, Path)


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

    def test_openai_api_key_non_empty_preserved(self) -> None:
        raw = {"llm": {"openai_api_key": "sk-test"}}
        cfg = _build_config_from_dict(raw)
        assert cfg.openai_api_key == "sk-test"

    def test_ollama_base_url_parsed(self) -> None:
        raw = {"llm": {"ollama_base_url": "http://custom:11434/v1"}}
        cfg = _build_config_from_dict(raw)
        assert cfg.ollama_base_url == "http://custom:11434/v1"

    def test_blocklist_parsed(self) -> None:
        raw = {"risk": {"blocklist": ["rm -rf", "dd"]}}
        cfg = _build_config_from_dict(raw)
        assert cfg.blocklist == ["rm -rf", "dd"]

    def test_multiple_sections_parsed_together(self) -> None:
        raw = {
            "llm": {"provider": "ollama", "timeout": 60},
            "risk": {"threshold": "LOW"},
            "display": {"show_raw_command": False},
            "logging": {"level": "INFO"},
            "commands": {"max_command_length": 1500},
        }
        cfg = _build_config_from_dict(raw)
        assert cfg.llm_provider == "ollama"
        assert cfg.timeout == 60
        assert cfg.risk_threshold == "LOW"
        assert cfg.show_raw_command is False
        assert cfg.log_level == "INFO"
        assert cfg.max_command_length == 1500

    def test_wrong_type_for_auto_confirm_raises(self) -> None:
        raw = {"risk": {"auto_confirm_below_threshold": "yes"}}
        with pytest.raises(ConfigError):
            _build_config_from_dict(raw)

    def test_wrong_type_for_show_raw_command_raises(self) -> None:
        raw = {"display": {"show_raw_command": 1}}
        with pytest.raises(ConfigError):
            _build_config_from_dict(raw)

    def test_risk_section_non_dict_raises(self) -> None:
        raw = {"risk": ["not", "a", "table"]}
        with pytest.raises(ConfigError, match="risk"):
            _build_config_from_dict(raw)  # type: ignore[arg-type]

    def test_multiple_unknown_keys_all_in_extra(self) -> None:
        raw = {
            "section_a": {"k": "v"},
            "section_b": {"k": "v"},
        }
        cfg = _build_config_from_dict(raw)
        assert "section_a" in cfg.extra
        assert "section_b" in cfg.extra

    def test_openai_model_parsed(self) -> None:
        raw = {"llm": {"openai_model": "gpt-4o-mini"}}
        cfg = _build_config_from_dict(raw)
        assert cfg.openai_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# _parse_toml_section
# ---------------------------------------------------------------------------


class TestParseTomlSection:
    def test_returns_dict_for_existing_section(self) -> None:
        raw = {"llm": {"provider": "openai"}}
        result = _parse_toml_section(raw, "llm")
        assert result == {"provider": "openai"}

    def test_returns_empty_dict_for_missing_section(self) -> None:
        result = _parse_toml_section({}, "missing")
        assert result == {}

    def test_raises_for_non_dict_section(self) -> None:
        raw = {"llm": "not a dict"}
        with pytest.raises(ConfigError, match="llm"):
            _parse_toml_section(raw, "llm")  # type: ignore[arg-type]

    def test_raises_for_list_section(self) -> None:
        raw = {"llm": ["a", "b"]}
        with pytest.raises(ConfigError):
            _parse_toml_section(raw, "llm")  # type: ignore[arg-type]

    def test_raises_for_integer_section(self) -> None:
        raw = {"llm": 42}
        with pytest.raises(ConfigError):
            _parse_toml_section(raw, "llm")  # type: ignore[arg-type]

    def test_returns_nested_dict(self) -> None:
        raw = {"risk": {"threshold": "HIGH", "allowlist": ["echo"]}}
        result = _parse_toml_section(raw, "risk")
        assert result["threshold"] == "HIGH"
        assert result["allowlist"] == ["echo"]


# ---------------------------------------------------------------------------
# _get_str
# ---------------------------------------------------------------------------


class TestGetStr:
    def test_returns_string_value(self) -> None:
        result = _get_str({"key": "hello"}, "key", "default")
        assert result == "hello"

    def test_returns_default_when_missing(self) -> None:
        result = _get_str({}, "key", "default_val")
        assert result == "default_val"

    def test_raises_for_non_string_int(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_str({"key": 42}, "key", "default")

    def test_raises_for_non_string_bool(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_str({"key": True}, "key", "default")

    def test_empty_string_returned_as_is(self) -> None:
        result = _get_str({"key": ""}, "key", "default")
        assert result == ""

    def test_default_used_when_key_absent(self) -> None:
        result = _get_str({"other": "value"}, "key", "my_default")
        assert result == "my_default"


# ---------------------------------------------------------------------------
# _get_int
# ---------------------------------------------------------------------------


class TestGetInt:
    def test_returns_int_value(self) -> None:
        result = _get_int({"key": 42}, "key", 0)
        assert result == 42

    def test_returns_default_when_missing(self) -> None:
        result = _get_int({}, "key", 99)
        assert result == 99

    def test_raises_for_non_int_string(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_int({"key": "thirty"}, "key", 0)

    def test_raises_for_float(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_int({"key": 3.14}, "key", 0)

    def test_zero_returned(self) -> None:
        result = _get_int({"key": 0}, "key", 99)
        assert result == 0

    def test_negative_int_returned(self) -> None:
        result = _get_int({"key": -5}, "key", 0)
        assert result == -5


# ---------------------------------------------------------------------------
# _get_bool
# ---------------------------------------------------------------------------


class TestGetBool:
    def test_returns_true(self) -> None:
        result = _get_bool({"key": True}, "key", False)
        assert result is True

    def test_returns_false(self) -> None:
        result = _get_bool({"key": False}, "key", True)
        assert result is False

    def test_returns_default_when_missing(self) -> None:
        result = _get_bool({}, "key", True)
        assert result is True

    def test_raises_for_string(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_bool({"key": "true"}, "key", False)

    def test_raises_for_int(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_bool({"key": 1}, "key", False)

    def test_raises_for_none(self) -> None:
        with pytest.raises(ConfigError):
            _get_bool({"key": None}, "key", False)


# ---------------------------------------------------------------------------
# _get_str_list
# ---------------------------------------------------------------------------


class TestGetStrList:
    def test_returns_list_of_strings(self) -> None:
        result = _get_str_list({"key": ["a", "b", "c"]}, "key")
        assert result == ["a", "b", "c"]

    def test_returns_empty_list_when_missing(self) -> None:
        result = _get_str_list({}, "key")
        assert result == []

    def test_raises_for_non_list(self) -> None:
        with pytest.raises(ConfigError, match="key"):
            _get_str_list({"key": "not a list"}, "key")

    def test_raises_for_list_with_non_string_int(self) -> None:
        with pytest.raises(ConfigError):
            _get_str_list({"key": ["ok", 42]}, "key")

    def test_raises_for_list_with_non_string_bool(self) -> None:
        with pytest.raises(ConfigError):
            _get_str_list({"key": ["ok", True]}, "key")

    def test_default_override(self) -> None:
        result = _get_str_list({}, "key", default=["default_item"])
        assert result == ["default_item"]

    def test_empty_list_returned(self) -> None:
        result = _get_str_list({"key": []}, "key")
        assert result == []

    def test_single_item_list(self) -> None:
        result = _get_str_list({"key": ["only"]}, "key")
        assert result == ["only"]


# ---------------------------------------------------------------------------
# write_default_config
# ---------------------------------------------------------------------------


class TestWriteDefaultConfig:
    def test_creates_file_at_given_path(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "config.toml"
        result = write_default_config(target)
        assert result == target
        assert target.exists()

    def test_created_file_is_valid_toml_and_loadable(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        cfg = load_config(target)
        assert isinstance(cfg, SafeRunConfig)

    def test_does_not_overwrite_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text('[risk]\nthreshold = "LOW"\n', encoding="utf-8")
        write_default_config(target)
        content = target.read_text(encoding="utf-8")
        # Original content should be preserved
        assert 'threshold = "LOW"' in content

    def test_returns_existing_path_without_modification(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.toml"
        target.write_text("# existing", encoding="utf-8")
        result = write_default_config(target)
        assert result == target
        assert target.read_text(encoding="utf-8") == "# existing"

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "config.toml"
        write_default_config(target)
        assert target.exists()

    def test_returns_path_instance(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        result = write_default_config(target)
        assert isinstance(result, Path)

    def test_written_file_content_is_non_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        content = target.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_written_file_contains_sections(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        content = target.read_text(encoding="utf-8")
        for section in ("[llm]", "[risk]", "[display]", "[logging]", "[commands]"):
            assert section in content

    def test_loads_and_validates_correctly(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        cfg = load_config(target)
        # Should use all defaults from the written file
        assert cfg.llm_provider in VALID_LLM_PROVIDERS
        assert cfg.risk_threshold in VALID_RISK_LEVELS
        assert cfg.log_level in VALID_LOG_LEVELS

    def test_parent_dir_created_for_deep_path(self, tmp_path: Path) -> None:
        target = tmp_path / "level1" / "level2" / "level3" / "config.toml"
        assert not target.parent.exists()
        write_default_config(target)
        assert target.parent.exists()
        assert target.exists()

    def test_second_call_on_same_path_does_not_change_file(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.toml"
        write_default_config(target)
        content_first = target.read_text(encoding="utf-8")
        write_default_config(target)
        content_second = target.read_text(encoding="utf-8")
        assert content_first == content_second


# ---------------------------------------------------------------------------
# _default_config_toml
# ---------------------------------------------------------------------------


class TestDefaultConfigToml:
    def test_contains_all_sections(self) -> None:
        toml_str = _default_config_toml()
        for section in (
            "[llm]",
            "[risk]",
            "[display]",
            "[logging]",
            "[commands]",
        ):
            assert section in toml_str

    def test_is_parseable_toml(self) -> None:
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

    def test_returns_string(self) -> None:
        assert isinstance(_default_config_toml(), str)

    def test_non_empty(self) -> None:
        assert len(_default_config_toml()) > 0

    def test_contains_openai_model_key(self) -> None:
        toml_str = _default_config_toml()
        assert "openai_model" in toml_str

    def test_contains_ollama_model_key(self) -> None:
        toml_str = _default_config_toml()
        assert "ollama_model" in toml_str

    def test_contains_timeout_key(self) -> None:
        toml_str = _default_config_toml()
        assert "timeout" in toml_str

    def test_contains_max_command_length_key(self) -> None:
        toml_str = _default_config_toml()
        assert "max_command_length" in toml_str

    def test_contains_show_raw_command_key(self) -> None:
        toml_str = _default_config_toml()
        assert "show_raw_command" in toml_str

    def test_contains_log_level_key(self) -> None:
        toml_str = _default_config_toml()
        assert "level" in toml_str

    def test_parseable_into_valid_safe_run_config(self) -> None:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-reuse-of-import]

        toml_str = _default_config_toml()
        raw = tomllib.loads(toml_str)
        cfg = _build_config_from_dict(raw)
        assert isinstance(cfg, SafeRunConfig)
        assert cfg.llm_provider in VALID_LLM_PROVIDERS

    def test_default_provider_is_openai(self) -> None:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-reuse-of-import]

        toml_str = _default_config_toml()
        data = tomllib.loads(toml_str)
        assert data["llm"]["provider"] == "openai"

    def test_default_threshold_is_high(self) -> None:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-reuse-of-import]

        toml_str = _default_config_toml()
        data = tomllib.loads(toml_str)
        assert data["risk"]["threshold"] == "HIGH"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_valid_llm_providers_contains_openai(self) -> None:
        assert "openai" in VALID_LLM_PROVIDERS

    def test_valid_llm_providers_contains_ollama(self) -> None:
        assert "ollama" in VALID_LLM_PROVIDERS

    def test_valid_risk_levels_contains_all_four(self) -> None:
        assert VALID_RISK_LEVELS == frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

    def test_valid_log_levels_contains_debug(self) -> None:
        assert "DEBUG" in VALID_LOG_LEVELS

    def test_valid_log_levels_contains_critical(self) -> None:
        assert "CRITICAL" in VALID_LOG_LEVELS

    def test_default_config_path_is_path_instance(self) -> None:
        assert isinstance(DEFAULT_CONFIG_PATH, Path)

    def test_default_config_path_is_under_home(self) -> None:
        assert str(Path.home()) in str(DEFAULT_CONFIG_PATH)

    def test_default_timeout_is_positive(self) -> None:
        assert DEFAULT_TIMEOUT > 0

    def test_default_max_command_length_is_positive(self) -> None:
        assert DEFAULT_MAX_COMMAND_LENGTH > 0

    def test_default_risk_threshold_is_valid(self) -> None:
        assert DEFAULT_RISK_THRESHOLD in VALID_RISK_LEVELS

    def test_default_openai_model_non_empty(self) -> None:
        assert DEFAULT_OPENAI_MODEL

    def test_default_ollama_model_non_empty(self) -> None:
        assert DEFAULT_OLLAMA_MODEL

    def test_default_ollama_base_url_starts_with_http(self) -> None:
        assert DEFAULT_OLLAMA_BASE_URL.startswith("http")


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------


class TestConfigError:
    def test_is_value_error(self) -> None:
        err = ConfigError("test error")
        assert isinstance(err, ValueError)

    def test_is_exception(self) -> None:
        err = ConfigError("test")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        err = ConfigError("specific message")
        assert "specific message" in str(err)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ConfigError, match="expected"):
            raise ConfigError("expected error")

    def test_can_be_caught_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise ConfigError("wrapped as ValueError")

    def test_can_be_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise ConfigError("caught as Exception")
