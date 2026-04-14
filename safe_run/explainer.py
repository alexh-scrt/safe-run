"""LLM explainer client for safe_run.

This module provides a unified interface for querying OpenAI GPT models or
local Ollama models (via its OpenAI-compatible API) to obtain plain-English
explanations of shell commands and structured risk assessments. It supports
graceful fallback from OpenAI to Ollama when the primary provider fails.

Typical usage example::

    from safe_run.config import load_config
    from safe_run.explainer import explain_command, ExplainerResult

    config = load_config()
    result = explain_command("rm -rf /tmp/old_project", config)
    print(result.explanation)
    print(result.llm_risk_level)
    print(result.provider_used)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError, AuthenticationError, OpenAIError

from safe_run.config import SafeRunConfig
from safe_run.risk import RiskLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a security-conscious shell command explainer. When given a shell command,
you must respond with a valid JSON object and nothing else — no markdown fences,
no prose before or after, just the raw JSON.

The JSON object must have exactly these keys:

{
  "explanation": "<plain-English explanation of what the command does, 1-3 sentences>",
  "risk_level": "<one of: LOW, MEDIUM, HIGH, CRITICAL>",
  "risk_reason": "<1-2 sentences explaining the risk assessment>",
  "effects": ["<short bullet describing one side-effect or action>"],
  "reversible": <true or false>
}

Guidelines for risk_level:
- LOW: Read-only, informational, or low-impact operations (ls, cat, echo, grep).
- MEDIUM: Operations with side effects that are generally recoverable
  (git reset --hard, pip install, wget, chmod without -R).
- HIGH: Potentially dangerous operations that may cause data loss, privilege
  escalation, or security exposure (sudo commands, chmod 777, iptables changes,
  system service stops, DROP DATABASE).
- CRITICAL: Irreversible or catastrophically destructive operations
  (rm -rf /, curl|bash, mkfs, dd to disk, fork bombs, reverse shells).

Be concise. The explanation must be readable by a developer who may not know
the command. Do not add warnings outside the JSON structure.
"""

_USER_PROMPT_TEMPLATE = "Explain this shell command: {command}"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExplainerResult:
    """Structured result returned by the LLM explainer.

    Attributes:
        explanation: Plain-English description of what the command does.
        llm_risk_level: Risk level assigned by the LLM (may differ from the
            rule-based pre-screener).
        risk_reason: The LLM's justification for its risk assessment.
        effects: List of short bullet-point descriptions of side effects.
        reversible: Whether the command's effects can be undone.
        provider_used: Which LLM provider was actually used (e.g. 'openai'
            or 'ollama').
        model_used: The specific model name that produced the result.
        raw_response: The raw JSON string returned by the LLM.
        error: If set, the explanation is a fallback and this field contains
            the error message that caused the fallback.
    """

    explanation: str
    llm_risk_level: RiskLevel
    risk_reason: str
    effects: list[str]
    reversible: bool
    provider_used: str
    model_used: str
    raw_response: str = ""
    error: Optional[str] = None

    @property
    def is_fallback(self) -> bool:
        """Return True if this result was produced by the fallback provider.

        Returns:
            True when the error field is set, indicating a fallback occurred.
        """
        return self.error is not None


# ---------------------------------------------------------------------------
# Fallback result factory
# ---------------------------------------------------------------------------


def _make_fallback_result(
    command: str,
    error_message: str,
    provider: str,
    model: str,
) -> ExplainerResult:
    """Create a safe fallback ExplainerResult when the LLM is unavailable.

    The fallback result contains a generic explanation derived from the command
    string and defers risk assessment to the rule-based pre-screener.

    Args:
        command: The original shell command.
        error_message: Description of what went wrong with the LLM call.
        provider: The provider that failed.
        model: The model that failed.

    Returns:
        An ExplainerResult with is_fallback == True.
    """
    truncated = command[:80] + "..." if len(command) > 80 else command
    return ExplainerResult(
        explanation=(
            f"LLM explanation unavailable. Command: `{truncated}`. "
            "Please review it manually before proceeding."
        ),
        llm_risk_level=RiskLevel.HIGH,
        risk_reason="Risk level elevated to HIGH because LLM explanation failed.",
        effects=["Unknown — LLM call failed"],
        reversible=False,
        provider_used=provider,
        model_used=model,
        raw_response="",
        error=error_message,
    )


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str, provider: str, model: str) -> ExplainerResult:
    """Parse a raw LLM JSON response string into an ExplainerResult.

    Tolerates minor formatting issues such as leading/trailing whitespace and
    code-fence wrappers that some models add despite explicit instructions.

    Args:
        raw: The raw text returned by the LLM.
        provider: The provider name (used to populate ExplainerResult).
        model: The model name (used to populate ExplainerResult).

    Returns:
        A populated ExplainerResult.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON or does
            not contain the expected keys.
    """
    # Strip optional markdown code fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}. Raw response: {raw!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(data).__name__}. Raw: {raw!r}"
        )

    # Validate and extract required fields
    explanation = _require_str(data, "explanation")
    risk_level_raw = _require_str(data, "risk_level").upper().strip()
    risk_reason = _require_str(data, "risk_reason")
    reversible = _extract_bool(data, "reversible")
    effects = _extract_str_list(data, "effects")

    # Map risk level string to enum, defaulting to HIGH on unknown values
    try:
        llm_risk_level = RiskLevel(risk_level_raw)
    except ValueError:
        logger.warning(
            "LLM returned unknown risk level %r; defaulting to HIGH",
            risk_level_raw,
        )
        llm_risk_level = RiskLevel.HIGH

    return ExplainerResult(
        explanation=explanation,
        llm_risk_level=llm_risk_level,
        risk_reason=risk_reason,
        effects=effects,
        reversible=reversible,
        provider_used=provider,
        model_used=model,
        raw_response=raw,
        error=None,
    )


def _require_str(data: dict, key: str) -> str:
    """Extract a non-empty string value from a dict, with a safe fallback.

    Args:
        data: The parsed JSON dict.
        key: The key to look up.

    Returns:
        The string value, or an empty string if missing/wrong type.
    """
    value = data.get(key, "")
    if isinstance(value, str):
        return value
    # Coerce to string if possible
    return str(value) if value is not None else ""


def _extract_bool(data: dict, key: str) -> bool:
    """Extract a boolean value from a dict, defaulting to False.

    Args:
        data: The parsed JSON dict.
        key: The key to look up.

    Returns:
        The boolean value or False if missing/wrong type.
    """
    value = data.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _extract_str_list(data: dict, key: str) -> list[str]:
    """Extract a list of strings from a dict, defaulting to empty list.

    Args:
        data: The parsed JSON dict.
        key: The key to look up.

    Returns:
        A list of strings.
    """
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


# ---------------------------------------------------------------------------
# OpenAI client builder
# ---------------------------------------------------------------------------


def _build_openai_client(config: SafeRunConfig) -> OpenAI:
    """Construct an OpenAI client configured for the OpenAI API.

    Args:
        config: The loaded SafeRunConfig.

    Returns:
        A configured OpenAI client instance.

    Raises:
        ExplainerError: If no API key is available.
    """
    api_key = config.effective_openai_api_key
    if not api_key:
        raise ExplainerError(
            "No OpenAI API key found. Set openai_api_key in config or "
            "the OPENAI_API_KEY environment variable."
        )
    http_client = httpx.Client(timeout=config.timeout)
    return OpenAI(
        api_key=api_key,
        http_client=http_client,
    )


def _build_ollama_client(config: SafeRunConfig) -> OpenAI:
    """Construct an OpenAI client pointed at the Ollama local API.

    Ollama exposes an OpenAI-compatible REST API, so we reuse the OpenAI
    client SDK with a custom base URL. A dummy API key is used since Ollama
    does not require authentication.

    Args:
        config: The loaded SafeRunConfig.

    Returns:
        An OpenAI client configured for the Ollama endpoint.
    """
    http_client = httpx.Client(timeout=config.timeout)
    return OpenAI(
        api_key="ollama",  # Ollama ignores the key but the client requires one
        base_url=config.ollama_base_url,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExplainerError(Exception):
    """Raised when the LLM explainer encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Core LLM call helper
# ---------------------------------------------------------------------------


def _call_llm(
    client: OpenAI,
    model: str,
    command: str,
    timeout: int,
) -> str:
    """Make a single LLM chat completion request and return the text content.

    Args:
        client: A configured OpenAI (or Ollama-compatible) client.
        model: The model name to request.
        command: The shell command to explain.
        timeout: Request timeout in seconds (passed via HTTP client).

    Returns:
        The raw text content of the first completion choice.

    Raises:
        APIConnectionError: If the API server cannot be reached.
        APITimeoutError: If the request times out.
        AuthenticationError: If the API key is invalid.
        OpenAIError: For other OpenAI API errors.
    """
    user_message = _USER_PROMPT_TEMPLATE.format(command=command)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Low temperature for more deterministic JSON output
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    return content


# ---------------------------------------------------------------------------
# Provider-specific explain functions
# ---------------------------------------------------------------------------


def _explain_with_openai(
    command: str,
    config: SafeRunConfig,
) -> ExplainerResult:
    """Request a command explanation from the OpenAI API.

    Args:
        command: The shell command to explain.
        config: The loaded SafeRunConfig with OpenAI settings.

    Returns:
        A populated ExplainerResult.

    Raises:
        ExplainerError: If the API call fails or the response cannot be parsed.
    """
    model = config.openai_model
    try:
        client = _build_openai_client(config)
        raw = _call_llm(client, model, command, config.timeout)
    except AuthenticationError as exc:
        raise ExplainerError(
            f"OpenAI authentication failed: {exc}"
        ) from exc
    except APITimeoutError as exc:
        raise ExplainerError(
            f"OpenAI request timed out after {config.timeout}s: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise ExplainerError(
            f"OpenAI connection error: {exc}"
        ) from exc
    except OpenAIError as exc:
        raise ExplainerError(
            f"OpenAI API error: {exc}"
        ) from exc

    try:
        return _parse_llm_response(raw, provider="openai", model=model)
    except ValueError as exc:
        raise ExplainerError(
            f"Failed to parse OpenAI response: {exc}"
        ) from exc


def _explain_with_ollama(
    command: str,
    config: SafeRunConfig,
) -> ExplainerResult:
    """Request a command explanation from a local Ollama instance.

    Args:
        command: The shell command to explain.
        config: The loaded SafeRunConfig with Ollama settings.

    Returns:
        A populated ExplainerResult.

    Raises:
        ExplainerError: If the Ollama API call fails or cannot be parsed.
    """
    model = config.ollama_model
    try:
        client = _build_ollama_client(config)
        raw = _call_llm(client, model, command, config.timeout)
    except APITimeoutError as exc:
        raise ExplainerError(
            f"Ollama request timed out after {config.timeout}s: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise ExplainerError(
            f"Cannot connect to Ollama at {config.ollama_base_url}: {exc}. "
            "Ensure Ollama is running locally."
        ) from exc
    except OpenAIError as exc:
        raise ExplainerError(
            f"Ollama API error: {exc}"
        ) from exc

    try:
        return _parse_llm_response(raw, provider="ollama", model=model)
    except ValueError as exc:
        raise ExplainerError(
            f"Failed to parse Ollama response: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_command(
    command: str,
    config: SafeRunConfig,
    *,
    allow_fallback: bool = True,
) -> ExplainerResult:
    """Explain a shell command using the configured LLM provider.

    Uses the provider specified in ``config.llm_provider`` as the primary
    source. If the primary provider fails and ``allow_fallback`` is True,
    automatically falls back to the other provider (OpenAI ↔ Ollama). If
    both providers fail, returns a safe fallback ExplainerResult rather than
    raising an exception.

    The command is truncated to ``config.max_command_length`` characters
    before being sent to the LLM.

    Args:
        command: The raw shell command string to explain.
        config: The loaded SafeRunConfig containing provider settings.
        allow_fallback: If True (default), try the other provider when the
            primary provider fails. If False, return a fallback result
            immediately on primary failure.

    Returns:
        An ExplainerResult. Check ``result.is_fallback`` to detect errors.
    """
    # Truncate command to the configured limit
    truncated_command = command[: config.max_command_length]
    if len(command) > config.max_command_length:
        logger.warning(
            "Command truncated from %d to %d characters for LLM submission.",
            len(command),
            config.max_command_length,
        )

    primary_provider = config.llm_provider
    fallback_provider = "ollama" if primary_provider == "openai" else "openai"

    # --- Attempt primary provider ---
    primary_error: Optional[str] = None
    try:
        result = _explain_with_provider(
            truncated_command, config, primary_provider
        )
        logger.debug(
            "Explanation obtained from %s model %s.",
            result.provider_used,
            result.model_used,
        )
        return result
    except ExplainerError as exc:
        primary_error = str(exc)
        logger.warning(
            "Primary LLM provider %r failed: %s",
            primary_provider,
            primary_error,
        )

    # --- Attempt fallback provider ---
    if allow_fallback:
        try:
            result = _explain_with_provider(
                truncated_command, config, fallback_provider
            )
            # Tag the result with the fallback error for transparency
            result_with_error = ExplainerResult(
                explanation=result.explanation,
                llm_risk_level=result.llm_risk_level,
                risk_reason=result.risk_reason,
                effects=result.effects,
                reversible=result.reversible,
                provider_used=result.provider_used,
                model_used=result.model_used,
                raw_response=result.raw_response,
                error=(
                    f"Primary provider {primary_provider!r} failed; "
                    f"used {fallback_provider!r} as fallback. "
                    f"Primary error: {primary_error}"
                ),
            )
            logger.info(
                "Explanation obtained from fallback provider %r.",
                fallback_provider,
            )
            return result_with_error
        except ExplainerError as fallback_exc:
            logger.warning(
                "Fallback LLM provider %r also failed: %s",
                fallback_provider,
                fallback_exc,
            )
            combined_error = (
                f"Primary ({primary_provider}) failed: {primary_error}; "
                f"fallback ({fallback_provider}) also failed: {fallback_exc}"
            )
            return _make_fallback_result(
                command=command,
                error_message=combined_error,
                provider=fallback_provider,
                model=(
                    config.ollama_model
                    if fallback_provider == "ollama"
                    else config.openai_model
                ),
            )

    # Fallback disabled — return degraded result
    return _make_fallback_result(
        command=command,
        error_message=primary_error or "Unknown error",
        provider=primary_provider,
        model=(
            config.openai_model
            if primary_provider == "openai"
            else config.ollama_model
        ),
    )


def _explain_with_provider(
    command: str,
    config: SafeRunConfig,
    provider: str,
) -> ExplainerResult:
    """Dispatch to the correct provider-specific explain function.

    Args:
        command: The (possibly truncated) shell command to explain.
        config: The loaded SafeRunConfig.
        provider: Either 'openai' or 'ollama'.

    Returns:
        An ExplainerResult from the chosen provider.

    Raises:
        ExplainerError: If the provider call fails.
        ValueError: If the provider name is unrecognised.
    """
    if provider == "openai":
        return _explain_with_openai(command, config)
    elif provider == "ollama":
        return _explain_with_ollama(command, config)
    else:
        raise ExplainerError(
            f"Unknown LLM provider {provider!r}. Must be 'openai' or 'ollama'."
        )


def combine_risk_levels(
    rule_level: RiskLevel,
    llm_level: RiskLevel,
) -> RiskLevel:
    """Return the higher of two risk levels for a conservative final assessment.

    The combined risk level takes the maximum of the rule-based pre-screener
    result and the LLM's assessment, ensuring neither source can downgrade
    a risk flagged by the other.

    Args:
        rule_level: Risk level from the rule-based pre-screener.
        llm_level: Risk level from the LLM explainer.

    Returns:
        The higher RiskLevel of the two inputs.
    """
    return rule_level if rule_level >= llm_level else llm_level
