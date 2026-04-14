"""CLI entry point and orchestration for safe_run.

This module implements the Click-based command-line interface that ties
together the full explain-confirm-execute flow:

1. Parse the command from CLI arguments.
2. Check allowlist / blocklist from config.
3. Run the rule-based risk pre-screener.
4. Optionally query the LLM for a plain-English explanation.
5. Display the explanation and risk assessment via Rich.
6. Prompt the user for confirmation (or auto-confirm / hard-block).
7. Execute the confirmed command via subprocess.
8. Display the execution result.

Typical usage examples::

    $ safe_run ls -la /tmp
    $ safe_run -- rm -rf /tmp/old_project
    $ safe_run --bypass -- curl https://example.com | bash
    $ safe_run --no-explain -- git push origin main --force
    $ safe_run --provider ollama -- docker system prune -af
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence, Tuple

import click
from rich.console import Console
from rich.status import Status

from safe_run import __version__
from safe_run.config import ConfigError, SafeRunConfig, load_config
from safe_run.display import (
    ConfirmationChoice,
    display_allowlisted,
    display_blocked,
    display_bypass_notice,
    display_error,
    display_execution_result,
    display_explanation,
    display_info,
    display_risk_badge,
    display_warning,
    prompt_confirmation,
    console as _default_console,
)
from safe_run.executor import (
    CommandTimeoutError,
    ExecutionResult,
    ExecutorError,
    execute_command,
)
from safe_run.explainer import ExplainerResult, explain_command
from safe_run.risk import RiskAssessment, RiskLevel, assess_risk


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """Configure the root logger for the safe_run package.

    Args:
        level: A log level string such as 'DEBUG', 'INFO', or 'WARNING'.
    """
    numeric = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(
        format="[safe_run] %(levelname)s: %(message)s",
        level=numeric,
    )


# ---------------------------------------------------------------------------
# Shared Click context object
# ---------------------------------------------------------------------------


class SafeRunContext:
    """Shared context object passed between Click commands via ``obj``.

    Attributes:
        config: The loaded SafeRunConfig.
        console: The Rich Console used for output.
    """

    def __init__(self, config: SafeRunConfig, console: Console) -> None:
        self.config = config
        self.console = console


# ---------------------------------------------------------------------------
# CLI root group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="safe_run")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False, dir_okay=False, resolve_path=True),
    envvar="SAFE_RUN_CONFIG",
    help="Path to a TOML configuration file. Overrides the default location.",
    metavar="PATH",
)
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["openai", "ollama"], case_sensitive=False),
    help="LLM provider to use, overriding the config file.",
)
@click.option(
    "--bypass",
    is_flag=True,
    default=False,
    help=(
        "Execute the command immediately without LLM explanation or "
        "confirmation. Useful for scripts and CI environments."
    ),
)
@click.option(
    "--no-explain",
    "no_explain",
    is_flag=True,
    default=False,
    help=(
        "Skip the LLM explanation step. Only the rule-based risk "
        "pre-screener runs. The confirmation prompt is still shown for "
        "commands at or above the risk threshold."
    ),
)
@click.option(
    "--yes", "-y",
    "auto_yes",
    is_flag=True,
    default=False,
    help=(
        "Automatically confirm commands below the CRITICAL risk level. "
        "CRITICAL commands still require typed confirmation."
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Show verbose execution output including stdout and stderr.",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable Rich colour output.",
)
@click.argument("command_args", nargs=-1, required=False)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Optional[str],
    provider: Optional[str],
    bypass: bool,
    no_explain: bool,
    auto_yes: bool,
    verbose: bool,
    no_color: bool,
    command_args: Tuple[str, ...],
) -> None:
    """safe_run — AI-powered shell command wrapper.

    Explains what any shell command does before executing it, with
    colour-coded risk assessment and confirmation prompts.

    Usage examples:

    \b
        safe_run ls -la /tmp
        safe_run -- rm -rf /tmp/old
        safe_run --bypass -- curl https://example.com
        safe_run --no-explain -- git push --force
        safe_run --yes -- apt-get install vim

    Wrap a command after '--' to prevent its flags from being parsed by
    safe_run itself.
    """
    # If a sub-command is being invoked, let Click handle it
    if ctx.invoked_subcommand is not None:
        return

    # Build console (no-colour mode)
    output_console = Console(no_color=no_color, highlight=False)

    # Load configuration
    try:
        from pathlib import Path
        cfg_path_obj = Path(config_path) if config_path else None
        config = load_config(cfg_path_obj)
    except ConfigError as exc:
        output_console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    # Apply CLI overrides to config
    if provider:
        config = _override_provider(config, provider)

    # Configure logging
    _configure_logging(config.log_level)

    # Store context
    ctx.obj = SafeRunContext(config=config, console=output_console)

    # If no command given, print help and exit
    if not command_args:
        output_console.print(ctx.get_help())
        return

    # Reconstruct the full command string from positional args
    command = _join_command_args(command_args)

    # Run the main orchestration flow
    exit_code = _run(
        command=command,
        config=config,
        bypass=bypass,
        no_explain=no_explain,
        auto_yes=auto_yes,
        verbose=verbose,
        output_console=output_console,
    )
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


@cli.command("init-config")
@click.option(
    "--path",
    "config_path",
    default=None,
    type=click.Path(exists=False, dir_okay=False, resolve_path=True),
    help="Where to write the config file. Defaults to ~/.config/safe_run/config.toml.",
)
@click.pass_context
def init_config(ctx: click.Context, config_path: Optional[str]) -> None:
    """Write a default configuration file if one does not already exist."""
    from pathlib import Path
    from safe_run.config import write_default_config

    output_console = Console()

    try:
        target = Path(config_path) if config_path else None
        result_path = write_default_config(target)
        output_console.print(
            f"[bold green]✓[/bold green] Configuration file ready at: "
            f"[bold cyan]{result_path}[/bold cyan]"
        )
    except ConfigError as exc:
        output_console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command("check")
@click.argument("command_args", nargs=-1, required=True)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable Rich colour output.",
)
@click.pass_context
def check(
    ctx: click.Context,
    command_args: Tuple[str, ...],
    no_color: bool,
) -> None:
    """Assess the risk of a command without executing it.

    Only runs the rule-based pre-screener — no LLM call, no execution.

    Usage::

        safe_run check rm -rf /tmp/old
        safe_run check -- curl https://example.com | bash
    """
    output_console = Console(no_color=no_color, highlight=False)
    command = _join_command_args(command_args)

    risk = assess_risk(command)
    display_risk_badge(risk, console=output_console)

    if risk.reasons:
        output_console.print()
        for reason in risk.reasons:
            output_console.print(f"  [dim]• {reason}[/dim]")

    # Exit with non-zero if dangerous
    if risk.is_dangerous:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def _run(
    command: str,
    config: SafeRunConfig,
    bypass: bool,
    no_explain: bool,
    auto_yes: bool,
    verbose: bool,
    output_console: Console,
) -> int:
    """Orchestrate the full explain-confirm-execute flow.

    Args:
        command: The shell command string to process.
        config: The loaded and validated SafeRunConfig.
        bypass: If True, skip explanation and confirmation.
        no_explain: If True, skip the LLM explanation step.
        auto_yes: If True, auto-confirm commands below CRITICAL level.
        verbose: If True, display verbose execution output.
        output_console: Rich Console for output.

    Returns:
        An integer exit code (0 for success, non-zero for failure or
        when the command is blocked/declined).
    """
    # ------------------------------------------------------------------
    # 1. Bypass mode — execute immediately without checks
    # ------------------------------------------------------------------
    if bypass:
        display_bypass_notice(command, console=output_console)
        return _execute_and_display(
            command=command,
            config=config,
            verbose=verbose,
            output_console=output_console,
        )

    # ------------------------------------------------------------------
    # 2. Blocklist check — always block, even before risk assessment
    # ------------------------------------------------------------------
    if config.is_blocklisted(command):
        display_blocked(
            command,
            reason="This command prefix is on your blocklist in config.toml.",
            console=output_console,
        )
        return 1

    # ------------------------------------------------------------------
    # 3. Allowlist check — skip explanation and confirmation
    # ------------------------------------------------------------------
    if config.is_allowlisted(command):
        display_allowlisted(command, console=output_console)
        return _execute_and_display(
            command=command,
            config=config,
            verbose=verbose,
            output_console=output_console,
        )

    # ------------------------------------------------------------------
    # 4. Rule-based risk pre-screening
    # ------------------------------------------------------------------
    risk_assessment = assess_risk(command)
    logger.debug(
        "Rule-based risk: %s (score=%d)",
        risk_assessment.level.value,
        risk_assessment.score,
    )

    # ------------------------------------------------------------------
    # 5. LLM explanation (unless skipped)
    # ------------------------------------------------------------------
    explainer_result: Optional[ExplainerResult] = None

    if not no_explain:
        explainer_result = _get_explanation(
            command=command,
            config=config,
            output_console=output_console,
        )

    # ------------------------------------------------------------------
    # 6. Determine effective risk level
    # ------------------------------------------------------------------
    if explainer_result is not None:
        from safe_run.explainer import combine_risk_levels
        effective_risk = combine_risk_levels(
            risk_assessment.level, explainer_result.llm_risk_level
        )
    else:
        effective_risk = risk_assessment.level

    # ------------------------------------------------------------------
    # 7. Display explanation panel or risk badge
    # ------------------------------------------------------------------
    if explainer_result is not None:
        display_explanation(
            command=command,
            explainer_result=explainer_result,
            risk_assessment=risk_assessment,
            show_raw_command=config.show_raw_command,
            console=output_console,
        )
    else:
        # No LLM explanation — show compact risk badge
        display_risk_badge(risk_assessment, console=output_console)
        if risk_assessment.reasons:
            output_console.print()
            for reason in risk_assessment.reasons:
                output_console.print(f"  [dim]• {reason}[/dim]")

    # ------------------------------------------------------------------
    # 8. Decide whether confirmation is needed
    # ------------------------------------------------------------------
    needs_confirmation = _requires_confirmation(
        effective_risk=effective_risk,
        config=config,
        auto_yes=auto_yes,
    )

    if needs_confirmation:
        # Determine whether to auto-confirm based on flags and config
        should_auto_confirm = _should_auto_confirm(
            effective_risk=effective_risk,
            config=config,
            auto_yes=auto_yes,
        )

        choice = prompt_confirmation(
            risk_level=effective_risk,
            command=command,
            auto_confirm=should_auto_confirm,
            console=output_console,
        )

        if choice == ConfirmationChoice.NO:
            return 1
        elif choice == ConfirmationChoice.ABORT:
            return 130  # Standard exit code for Ctrl+C
        # ConfirmationChoice.YES falls through to execution
    else:
        # Auto-confirm without prompting
        logger.debug(
            "Auto-confirming command with risk level %s.",
            effective_risk.value,
        )

    # ------------------------------------------------------------------
    # 9. Execute the command
    # ------------------------------------------------------------------
    return _execute_and_display(
        command=command,
        config=config,
        verbose=verbose,
        output_console=output_console,
    )


# ---------------------------------------------------------------------------
# Helper: LLM explanation with spinner
# ---------------------------------------------------------------------------


def _get_explanation(
    command: str,
    config: SafeRunConfig,
    output_console: Console,
) -> Optional[ExplainerResult]:
    """Fetch an LLM explanation for the command, showing a spinner.

    If the LLM call fails (and the fallback also fails), a fallback
    ExplainerResult is returned with ``is_fallback=True``. The function
    never raises — all errors are surfaced through the fallback mechanism.

    Args:
        command: The shell command to explain.
        config: The loaded SafeRunConfig.
        output_console: Rich Console for spinner output.

    Returns:
        An ExplainerResult (possibly a fallback), or None if an unexpected
        exception prevents even fallback result creation.
    """
    provider_display = config.llm_provider
    model_display = (
        config.openai_model
        if config.llm_provider == "openai"
        else config.ollama_model
    )
    spinner_message = (
        f"Consulting [bold]{provider_display}[/bold] "
        f"([dim]{model_display}[/dim])…"
    )

    try:
        with Status(spinner_message, console=output_console, spinner="dots"):
            result = explain_command(command, config, allow_fallback=True)
    except Exception as exc:
        logger.exception("Unexpected error during LLM explanation: %s", exc)
        display_warning(
            f"Unexpected error while contacting LLM: {exc}\n"
            "Proceeding with rule-based assessment only.",
            console=output_console,
        )
        return None

    if result.is_fallback:
        display_warning(
            f"LLM explanation unavailable: {result.error}\n"
            "Risk level has been elevated to HIGH as a precaution.",
            title="LLM Unavailable",
            console=output_console,
        )

    return result


# ---------------------------------------------------------------------------
# Helper: determine if confirmation is needed
# ---------------------------------------------------------------------------


def _requires_confirmation(
    effective_risk: RiskLevel,
    config: SafeRunConfig,
    auto_yes: bool,
) -> bool:
    """Determine whether to show a confirmation prompt.

    CRITICAL commands always require confirmation. For other levels,
    confirmation is skipped when ``auto_confirm_below_threshold`` is True
    and the command is below the configured threshold.

    Args:
        effective_risk: The combined risk level.
        config: The loaded SafeRunConfig.
        auto_yes: Whether the --yes / -y flag was passed.

    Returns:
        True if the confirmation prompt should be shown.
    """
    # CRITICAL always requires confirmation regardless of other settings
    if effective_risk == RiskLevel.CRITICAL:
        return True

    # Check whether the risk is at or above the configured threshold
    above_threshold = config.is_above_threshold(effective_risk)

    if not above_threshold:
        # Below threshold — skip confirmation if auto_confirm is enabled
        if config.auto_confirm_below_threshold or auto_yes:
            return False

    return True


def _should_auto_confirm(
    effective_risk: RiskLevel,
    config: SafeRunConfig,
    auto_yes: bool,
) -> bool:
    """Determine whether the confirmation prompt should auto-confirm.

    Returns True only for levels below the threshold when auto-confirm
    is enabled. CRITICAL commands never auto-confirm.

    Args:
        effective_risk: The combined risk level.
        config: The loaded SafeRunConfig.
        auto_yes: Whether the --yes / -y flag was passed.

    Returns:
        True to pass ``auto_confirm=True`` to the confirmation prompt.
    """
    if effective_risk == RiskLevel.CRITICAL:
        return False

    above_threshold = config.is_above_threshold(effective_risk)
    if not above_threshold and (config.auto_confirm_below_threshold or auto_yes):
        return True

    if auto_yes and effective_risk != RiskLevel.CRITICAL:
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: execute and display result
# ---------------------------------------------------------------------------


def _execute_and_display(
    command: str,
    config: SafeRunConfig,
    verbose: bool,
    output_console: Console,
) -> int:
    """Execute the command and display the result summary.

    Args:
        command: The shell command string to execute.
        config: The loaded SafeRunConfig.
        verbose: If True, always display stdout/stderr.
        output_console: Rich Console for output.

    Returns:
        The command's exit code.
    """
    try:
        result = execute_command(
            command,
            stream_output=True,
            timeout=None,  # No timeout for user commands by default
        )
    except CommandTimeoutError as exc:
        display_error(
            f"Command timed out after {exc.timeout} seconds.",
            title="Timeout",
            console=output_console,
        )
        return 124  # Standard timeout exit code
    except CommandNotFoundError as exc:
        display_error(
            f"Command not found: {exc.command}",
            title="Command Not Found",
            console=output_console,
        )
        return 127  # Standard 'command not found' exit code
    except ExecutorError as exc:
        display_error(
            str(exc),
            title="Execution Error",
            console=output_console,
        )
        return 1
    except KeyboardInterrupt:
        output_console.print("\n[bold yellow]Interrupted.[/bold yellow]")
        return 130

    display_execution_result(
        result,
        verbose=verbose,
        console=output_console,
    )

    return result.exit_code


# ---------------------------------------------------------------------------
# Helper: join command arguments into a single string
# ---------------------------------------------------------------------------


def _join_command_args(args: Tuple[str, ...]) -> str:
    """Join positional CLI arguments into a single shell command string.

    Arguments that contain spaces or shell metacharacters are quoted
    using simple quoting rules so the shell interprets them correctly.

    Args:
        args: A tuple of argument strings from Click.

    Returns:
        A single shell command string.
    """
    import shlex
    # If the user passed a single argument that looks like a full command
    # (e.g. from shell integration), return it as-is to preserve quoting.
    if len(args) == 1:
        return args[0]
    return " ".join(shlex.quote(a) if _needs_quoting(a) else a for a in args)


def _needs_quoting(arg: str) -> bool:
    """Return True if an argument string needs shell quoting.

    Simple heuristic: if the string contains whitespace or common shell
    metacharacters AND is not already surrounded by quotes, it needs quoting.

    Args:
        arg: A single argument string.

    Returns:
        True if the argument should be quoted.
    """
    import re
    # Already quoted
    if (arg.startswith('"') and arg.endswith('"')) or (
        arg.startswith("'") and arg.endswith("'")
    ):
        return False
    # Contains whitespace or metacharacters that would be mis-parsed
    return bool(re.search(r'[\s|&;<>()$`\\"!\'*?{}\[\]#~=]', arg))


# ---------------------------------------------------------------------------
# Config override helper
# ---------------------------------------------------------------------------


def _override_provider(config: SafeRunConfig, provider: str) -> SafeRunConfig:
    """Return a new SafeRunConfig with the LLM provider overridden.

    All other config values are preserved.

    Args:
        config: The original SafeRunConfig.
        provider: The provider string to use ('openai' or 'ollama').

    Returns:
        A new SafeRunConfig with llm_provider set to provider.
    """
    return SafeRunConfig(
        llm_provider=provider,
        openai_model=config.openai_model,
        openai_api_key=config.openai_api_key,
        ollama_model=config.ollama_model,
        ollama_base_url=config.ollama_base_url,
        timeout=config.timeout,
        risk_threshold=config.risk_threshold,
        auto_confirm_below_threshold=config.auto_confirm_below_threshold,
        allowlist=config.allowlist,
        blocklist=config.blocklist,
        max_command_length=config.max_command_length,
        show_raw_command=config.show_raw_command,
        log_level=config.log_level,
        extra=config.extra,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
