"""Configuration loader for safe_run.

This module handles loading, validating, and providing defaults for
configuration stored in ~/.config/safe_run/config.toml. It supports
LLM provider selection, model specification, risk thresholds, and
per-command allowlists.

Typical usage example::

    from safe_run.config import load_config, SafeRunConfig

    config = load_config()
    print(config.llm_provider)  # 'openai'
    print(config.risk_threshold)  # 'HIGH'
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-reuse-of-import]
    except ImportError as exc:
        raise ImportError(
            "tomli is required for Python < 3.11. "
            "Install it with: pip install tomli"
        ) from exc


# Default configuration file path
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "safe_run"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"

# Valid values for constrained config fields
VALID_LLM_PROVIDERS = frozenset({"openai", "ollama"})
VALID_RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Default OpenAI model
DEFAULT_OPENAI_MODEL = "gpt-4o"
# Default Ollama model
DEFAULT_OLLAMA_MODEL = "llama3"
# Default Ollama API base URL
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Default request timeout in seconds
DEFAULT_TIMEOUT = 30
# Default risk threshold above which confirmation is required
DEFAULT_RISK_THRESHOLD = "HIGH"
# Default maximum command length to send to LLM
DEFAULT_MAX_COMMAND_LENGTH = 2000


class ConfigError(ValueError):
    """Raised when the configuration file contains invalid or missing values."""


@dataclass
class SafeRunConfig:
    """Holds the validated runtime configuration for safe_run.

    Attributes:
        llm_provider: Which LLM backend to use. Either 'openai' or 'ollama'.
        openai_model: OpenAI model name, e.g. 'gpt-4o'.
        openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        ollama_model: Ollama model name, e.g. 'llama3'.
        ollama_base_url: Base URL for the Ollama OpenAI-compatible API.
        timeout: HTTP request timeout in seconds for LLM calls.
        risk_threshold: Minimum risk level requiring explicit confirmation.
            One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
        auto_confirm_below_threshold: If True, automatically approve commands
            below the risk threshold without prompting.
        allowlist: List of command prefixes that are always approved without
            LLM explanation or confirmation.
        blocklist: List of command prefixes that are always blocked.
        max_command_length: Maximum number of characters to send to the LLM.
        show_raw_command: If True, display the raw command in the UI panel.
        log_level: Logging verbosity. One of DEBUG/INFO/WARNING/ERROR/CRITICAL.
        extra: Dictionary of any unrecognised keys for forward compatibility.
    """

    llm_provider: str = "openai"
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_api_key: Optional[str] = None
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    risk_threshold: str = DEFAULT_RISK_THRESHOLD
    auto_confirm_below_threshold: bool = True
    allowlist: List[str] = field(default_factory=list)
    blocklist: List[str] = field(default_factory=list)
    max_command_length: int = DEFAULT_MAX_COMMAND_LENGTH
    show_raw_command: bool = True
    log_level: str = "WARNING"
    extra: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field values after initialisation.

        Raises:
            ConfigError: If any field has an invalid value.
        """
        self._validate()

    def _validate(self) -> None:
        """Perform all validation checks.

        Raises:
            ConfigError: If any field has an invalid value.
        """
        if self.llm_provider not in VALID_LLM_PROVIDERS:
            raise ConfigError(
                f"Invalid llm_provider {self.llm_provider!r}. "
                f"Must be one of: {sorted(VALID_LLM_PROVIDERS)}"
            )

        if self.risk_threshold not in VALID_RISK_LEVELS:
            raise ConfigError(
                f"Invalid risk_threshold {self.risk_threshold!r}. "
                f"Must be one of: {sorted(VALID_RISK_LEVELS)}"
            )

        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"Invalid log_level {self.log_level!r}. "
                f"Must be one of: {sorted(VALID_LOG_LEVELS)}"
            )

        if self.timeout <= 0:
            raise ConfigError(
                f"timeout must be a positive integer, got {self.timeout!r}"
            )

        if self.max_command_length <= 0:
            raise ConfigError(
                f"max_command_length must be a positive integer, "
                f"got {self.max_command_length!r}"
            )

        if not isinstance(self.allowlist, list):
            raise ConfigError(
                f"allowlist must be a list of strings, got {type(self.allowlist)!r}"
            )

        if not isinstance(self.blocklist, list):
            raise ConfigError(
                f"blocklist must be a list of strings, got {type(self.blocklist)!r}"
            )

        for item in self.allowlist:
            if not isinstance(item, str):
                raise ConfigError(
                    f"All allowlist entries must be strings, got {item!r}"
                )

        for item in self.blocklist:
            if not isinstance(item, str):
                raise ConfigError(
                    f"All blocklist entries must be strings, got {item!r}"
                )

    @property
    def effective_openai_api_key(self) -> Optional[str]:
        """Return the effective OpenAI API key.

        Checks the config value first, then falls back to the
        OPENAI_API_KEY environment variable.

        Returns:
            The API key string or None if not configured.
        """
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def risk_level_order(self) -> List[str]:
        """Return risk levels in ascending severity order.

        Returns:
            List of risk level strings from lowest to highest severity.
        """
        return ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def is_above_threshold(self, risk_level: str) -> bool:
        """Return True if the given risk level meets or exceeds the threshold.

        Args:
            risk_level: A risk level string such as 'LOW', 'MEDIUM', 'HIGH',
                or 'CRITICAL'.

        Returns:
            True if risk_level is at or above self.risk_threshold.

        Raises:
            ConfigError: If risk_level is not a valid risk level string.
        """
        if risk_level not in VALID_RISK_LEVELS:
            raise ConfigError(
                f"Unknown risk level {risk_level!r}. "
                f"Must be one of: {sorted(VALID_RISK_LEVELS)}"
            )
        order = self.risk_level_order
        return order.index(risk_level) >= order.index(self.risk_threshold)

    def is_allowlisted(self, command: str) -> bool:
        """Check whether a command matches any allowlist prefix.

        Args:
            command: The full shell command string to check.

        Returns:
            True if the command starts with any allowlist entry.
        """
        stripped = command.strip()
        return any(stripped.startswith(prefix) for prefix in self.allowlist)

    def is_blocklisted(self, command: str) -> bool:
        """Check whether a command matches any blocklist prefix.

        Args:
            command: The full shell command string to check.

        Returns:
            True if the command starts with any blocklist entry.
        """
        stripped = command.strip()
        return any(stripped.startswith(prefix) for prefix in self.blocklist)


def _parse_toml_section(
    raw: Dict[str, object],
    section: str,
) -> Dict[str, object]:
    """Safely extract a sub-table from a parsed TOML document.

    Args:
        raw: The top-level parsed TOML dictionary.
        section: The key for the sub-table to extract.

    Returns:
        The sub-table dict, or an empty dict if the section is absent.

    Raises:
        ConfigError: If the section key exists but is not a dict.
    """
    value = raw.get(section, {})
    if not isinstance(value, dict):
        raise ConfigError(
            f"Configuration section [{section}] must be a TOML table, "
            f"got {type(value).__name__!r}"
        )
    return value  # type: ignore[return-value]


def _get_str(
    data: Dict[str, object],
    key: str,
    default: str,
) -> str:
    """Extract a string value from a config dict with a default.

    Args:
        data: Dictionary of config key-value pairs.
        key: The config key to look up.
        default: The default string value to return if key is absent.

    Returns:
        The string value from data, or default.

    Raises:
        ConfigError: If the value is present but not a string.
    """
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(
            f"Configuration key {key!r} must be a string, "
            f"got {type(value).__name__!r}: {value!r}"
        )
    return value


def _get_int(
    data: Dict[str, object],
    key: str,
    default: int,
) -> int:
    """Extract an integer value from a config dict with a default.

    Args:
        data: Dictionary of config key-value pairs.
        key: The config key to look up.
        default: The default integer value to return if key is absent.

    Returns:
        The integer value from data, or default.

    Raises:
        ConfigError: If the value is present but not an integer.
    """
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ConfigError(
            f"Configuration key {key!r} must be an integer, "
            f"got {type(value).__name__!r}: {value!r}"
        )
    return value


def _get_bool(
    data: Dict[str, object],
    key: str,
    default: bool,
) -> bool:
    """Extract a boolean value from a config dict with a default.

    Args:
        data: Dictionary of config key-value pairs.
        key: The config key to look up.
        default: The default boolean value to return if key is absent.

    Returns:
        The boolean value from data, or default.

    Raises:
        ConfigError: If the value is present but not a boolean.
    """
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"Configuration key {key!r} must be a boolean, "
            f"got {type(value).__name__!r}: {value!r}"
        )
    return value


def _get_str_list(
    data: Dict[str, object],
    key: str,
    default: Optional[List[str]] = None,
) -> List[str]:
    """Extract a list of strings from a config dict with a default.

    Args:
        data: Dictionary of config key-value pairs.
        key: The config key to look up.
        default: The default list to return if key is absent. Defaults to [].

    Returns:
        The list of strings from data, or default.

    Raises:
        ConfigError: If the value is present but not a list of strings.
    """
    if default is None:
        default = []
    value = data.get(key, default)
    if not isinstance(value, list):
        raise ConfigError(
            f"Configuration key {key!r} must be an array, "
            f"got {type(value).__name__!r}"
        )
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(
                f"All entries in {key!r} must be strings, got {item!r}"
            )
    return value  # type: ignore[return-value]


def _build_config_from_dict(raw: Dict[str, object]) -> SafeRunConfig:
    """Build a SafeRunConfig from a raw parsed TOML dictionary.

    Understands the following TOML structure::

        [llm]
        provider = "openai"
        openai_model = "gpt-4o"
        openai_api_key = "sk-..."
        ollama_model = "llama3"
        ollama_base_url = "http://localhost:11434/v1"
        timeout = 30

        [risk]
        threshold = "HIGH"
        auto_confirm_below_threshold = true
        allowlist = ["echo", "ls", "cat"]
        blocklist = []

        [display]
        show_raw_command = true

        [logging]
        level = "WARNING"

        [commands]
        max_command_length = 2000

    Args:
        raw: Top-level dictionary from tomllib.loads() / tomllib.load().

    Returns:
        A validated SafeRunConfig instance.

    Raises:
        ConfigError: If any required value has an invalid type or value.
    """
    llm = _parse_toml_section(raw, "llm")
    risk = _parse_toml_section(raw, "risk")
    display = _parse_toml_section(raw, "display")
    logging_section = _parse_toml_section(raw, "logging")
    commands = _parse_toml_section(raw, "commands")

    # Collect unknown top-level keys for forward compatibility
    known_sections = {"llm", "risk", "display", "logging", "commands"}
    extra: Dict[str, object] = {
        k: v for k, v in raw.items() if k not in known_sections
    }

    return SafeRunConfig(
        llm_provider=_get_str(llm, "provider", "openai"),
        openai_model=_get_str(llm, "openai_model", DEFAULT_OPENAI_MODEL),
        openai_api_key=(
            _get_str(llm, "openai_api_key", "") or None
        ),
        ollama_model=_get_str(llm, "ollama_model", DEFAULT_OLLAMA_MODEL),
        ollama_base_url=_get_str(llm, "ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
        timeout=_get_int(llm, "timeout", DEFAULT_TIMEOUT),
        risk_threshold=_get_str(risk, "threshold", DEFAULT_RISK_THRESHOLD),
        auto_confirm_below_threshold=_get_bool(
            risk, "auto_confirm_below_threshold", True
        ),
        allowlist=_get_str_list(risk, "allowlist"),
        blocklist=_get_str_list(risk, "blocklist"),
        max_command_length=_get_int(
            commands, "max_command_length", DEFAULT_MAX_COMMAND_LENGTH
        ),
        show_raw_command=_get_bool(display, "show_raw_command", True),
        log_level=_get_str(logging_section, "level", "WARNING"),
        extra=extra,
    )


def load_config(
    config_path: Optional[Path] = None,
) -> SafeRunConfig:
    """Load and return the safe_run configuration.

    If a config file is found at config_path (or the default location
    ~/.config/safe_run/config.toml), it is parsed and merged with defaults.
    If no file exists, a default SafeRunConfig is returned without error.

    The SAFE_RUN_CONFIG environment variable can override the config path.

    Args:
        config_path: Optional explicit path to a TOML config file. If None,
            the SAFE_RUN_CONFIG env var is checked, then the default path
            ~/.config/safe_run/config.toml is used.

    Returns:
        A validated SafeRunConfig populated from the file and/or defaults.

    Raises:
        ConfigError: If the config file exists but contains invalid TOML
            syntax, wrong value types, or values that fail validation.
    """
    resolved_path = _resolve_config_path(config_path)

    if resolved_path is None or not resolved_path.exists():
        return SafeRunConfig()

    try:
        with resolved_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(
            f"Cannot read configuration file {resolved_path}: {exc}"
        ) from exc
    except Exception as exc:
        # tomllib raises TOMLDecodeError (a subclass of ValueError)
        raise ConfigError(
            f"Invalid TOML in configuration file {resolved_path}: {exc}"
        ) from exc

    return _build_config_from_dict(raw)


def _resolve_config_path(explicit: Optional[Path]) -> Optional[Path]:
    """Determine which config file path to use.

    Priority order:
    1. The explicit path argument (if given).
    2. The SAFE_RUN_CONFIG environment variable (if set).
    3. The default ~/.config/safe_run/config.toml.

    Args:
        explicit: An explicitly provided Path, or None.

    Returns:
        The resolved Path to check, or None if no path is determinable.
    """
    if explicit is not None:
        return explicit

    env_path = os.environ.get("SAFE_RUN_CONFIG")
    if env_path:
        return Path(env_path)

    return DEFAULT_CONFIG_PATH


def write_default_config(config_path: Optional[Path] = None) -> Path:
    """Write a default configuration file if one does not already exist.

    Creates the parent directory if needed. Does not overwrite an existing file.

    Args:
        config_path: Where to write the config file. Defaults to
            ~/.config/safe_run/config.toml.

    Returns:
        The path to the (possibly newly created) config file.

    Raises:
        ConfigError: If the directory cannot be created or the file cannot
            be written.
    """
    target = config_path or DEFAULT_CONFIG_PATH

    if target.exists():
        return target

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(
            f"Cannot create configuration directory {target.parent}: {exc}"
        ) from exc

    default_toml = _default_config_toml()

    try:
        target.write_text(default_toml, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Cannot write configuration file {target}: {exc}"
        ) from exc

    return target


def _default_config_toml() -> str:
    """Return a TOML string representing the default safe_run configuration.

    Returns:
        A multi-line TOML string with all supported keys documented.
    """
    return """\
# safe_run configuration file
# Location: ~/.config/safe_run/config.toml

[llm]
# LLM provider to use: "openai" or "ollama"
provider = "openai"

# OpenAI model name (used when provider = "openai")
openai_model = "gpt-4o"

# OpenAI API key — if omitted, falls back to OPENAI_API_KEY env var
# openai_api_key = "sk-..."

# Ollama model name (used when provider = "ollama")
ollama_model = "llama3"

# Ollama API base URL (OpenAI-compatible endpoint)
ollama_base_url = "http://localhost:11434/v1"

# HTTP request timeout in seconds for LLM API calls
timeout = 30

[risk]
# Minimum risk level that requires explicit user confirmation.
# One of: LOW, MEDIUM, HIGH, CRITICAL
threshold = "HIGH"

# If true, commands below the threshold are auto-approved without prompting
auto_confirm_below_threshold = true

# Commands that always pass without LLM explanation or confirmation.
# Matched as prefix of the stripped command string.
allowlist = []

# Commands that are always blocked and never executed.
blocklist = []

[commands]
# Maximum number of characters in a command sent to the LLM
max_command_length = 2000

[display]
# Show the raw command string in the explanation panel
show_raw_command = true

[logging]
# Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = "WARNING"
"""
