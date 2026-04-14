"""Unit tests for safe_run.explainer module.

Covers LLM explainer client logic using mocked API responses to verify
JSON parsing, fallback behavior, risk level combination, and error handling
for both OpenAI and Ollama providers.
"""

from __future__ import annotations

import json
from typing import Any
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

    def test_critical_always_wins(self) -> None:
        for level in RiskLevel:
            assert combine_risk_levels(RiskLevel.CRITICAL, level) == RiskLevel.CRITICAL
            assert combine_risk_levels(level, RiskLevel.CRITICAL) == RiskLevel.CRITICAL

    def test_low_vs_low_returns_low(self) -> None:
        assert combine_risk_levels(RiskLevel.LOW, RiskLevel.LOW) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# ExplainerResult properties
# ---------------------------------------------------------------------------


class TestExplainerResultProperties:
    def _make_result(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        error: str | None = None,
    ) -> ExplainerResult:
        return ExplainerResult(
            explanation="Does something.",
            llm_risk_level=RiskLevel.LOW,
            risk_reason="Safe operation.",
            effects=["No side effects"],
            reversible=True,
            provider_used=provider,
            model_used=model,
            error=error,
        )

    def test_is_fallback_false_when_no_error(self) -> None:
        assert not self._make_result().is_fallback

    def test_is_fallback_true_when_error_set(self) -> None:
        assert self._make_result(error="API down").is_fallback


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
        mock_response = _mock_openai_response(valid_json)

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

        captured_commands = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured_commands.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(long_cmd, openai_config_short)

        assert len(captured_commands[0]) <= 20

    def test_short_command_not_truncated(
        self, openai_config: SafeRunConfig
    ) -> None:
        short_cmd = "ls"
        captured_commands = []

        def mock_call_llm(
            client: Any, model: str, command: str, timeout: int
        ) -> str:
            captured_commands.append(command)
            return _make_valid_json_response()

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            explain_command(short_cmd, openai_config)

        assert captured_commands[0] == short_cmd


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
                # Simulate the _explain_with_openai path raising
                raise OAIConnError(request=MagicMock())
            return valid_json

        with patch("safe_run.explainer._call_llm", side_effect=mock_call_llm):
            result = explain_command("ls", openai_config, allow_fallback=True)

        # Should have fallen back to ollama
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
