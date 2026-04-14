"""Rich-powered terminal UI for safe_run.

This module provides the terminal user interface components for safe_run,
including explanation panels with syntax highlighting, color-coded risk
badges, interactive confirmation prompts, and execution result displays.

All output is rendered using the Rich library for a polished, consistent
terminal experience.

Typical usage example::

    from safe_run.display import (
        display_explanation,
        prompt_confirmation,
        display_execution_result,
        ConfirmationChoice,
    )
    from safe_run.risk import RiskLevel

    display_explanation(command, explainer_result, risk_assessment)
    choice = prompt_confirmation(risk_level=RiskLevel.HIGH)
    if choice == ConfirmationChoice.YES:
        ...
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from safe_run.executor import ExecutionResult
from safe_run.explainer import ExplainerResult
from safe_run.risk import RiskAssessment, RiskLevel, get_risk_level_color, get_risk_level_emoji


# ---------------------------------------------------------------------------
# Rich console configuration
# ---------------------------------------------------------------------------

_SAFE_RUN_THEME = Theme(
    {
        "safe_run.low": "bold green",
        "safe_run.medium": "bold yellow",
        "safe_run.high": "bold orange3",
        "safe_run.critical": "bold red",
        "safe_run.command": "bold cyan",
        "safe_run.label": "dim",
        "safe_run.success": "bold green",
        "safe_run.failure": "bold red",
        "safe_run.warning": "bold yellow",
        "safe_run.info": "bold blue",
        "safe_run.provider": "dim italic",
    }
)

# Module-level console instance; can be replaced in tests
console = Console(theme=_SAFE_RUN_THEME, highlight=False)


# ---------------------------------------------------------------------------
# Confirmation choices
# ---------------------------------------------------------------------------


class ConfirmationChoice(Enum):
    """Possible responses to the confirmation prompt.

    Attributes:
        YES: The user confirmed they want to run the command.
        NO: The user declined to run the command.
        ABORT: The user force-aborted (e.g. Ctrl+C or 'abort' input).
    """

    YES = "yes"
    NO = "no"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Risk level styling helpers
# ---------------------------------------------------------------------------


def _risk_style(level: RiskLevel) -> str:
    """Return the Rich theme key for a given risk level.

    Args:
        level: A RiskLevel enum value.

    Returns:
        A theme style name string.
    """
    return f"safe_run.{level.value.lower()}"


def _risk_badge(level: RiskLevel) -> Text:
    """Build a Rich Text object representing a colored risk badge.

    Args:
        level: A RiskLevel enum value.

    Returns:
        A styled Rich Text object like " ⚠️  HIGH ".
    """
    emoji = get_risk_level_emoji(level)
    label = level.value
    color = get_risk_level_color(level)

    text = Text()
    text.append(f" {emoji}  ", style=color)
    text.append(f"{label} ", style=f"bold {color}")
    return text


def _risk_panel_border_style(level: RiskLevel) -> str:
    """Return the panel border color for a given risk level.

    Args:
        level: A RiskLevel enum value.

    Returns:
        A Rich color string suitable for panel border_style.
    """
    color_map = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "orange3",
        RiskLevel.CRITICAL: "red",
    }
    return color_map.get(level, "white")


# ---------------------------------------------------------------------------
# Explanation panel
# ---------------------------------------------------------------------------


def display_explanation(
    command: str,
    explainer_result: ExplainerResult,
    risk_assessment: RiskAssessment,
    *,
    show_raw_command: bool = True,
    console: Console = console,
) -> None:
    """Render the full explanation panel to the terminal.

    Displays the command (optionally), the LLM explanation, risk badge,
    risk reasons, and command effects in a formatted Rich panel.

    Args:
        command: The original shell command string.
        explainer_result: The structured result from the LLM explainer.
        risk_assessment: The result from the rule-based risk pre-screener.
        show_raw_command: If True, display the command string in the panel
            header.
        console: Rich Console to write to. Defaults to module console.
    """
    # Determine the dominant (highest) risk level between rule-based and LLM
    from safe_run.explainer import combine_risk_levels
    combined_level = combine_risk_levels(
        risk_assessment.level, explainer_result.llm_risk_level
    )

    border_style = _risk_panel_border_style(combined_level)

    # Build panel content
    content = Text()

    # Command display
    if show_raw_command:
        content.append("Command: ", style="safe_run.label")
        content.append(escape(command), style="safe_run.command")
        content.append("\n\n")

    # LLM explanation text
    content.append("What it does:\n", style="bold")
    content.append(explainer_result.explanation)
    content.append("\n")

    # Risk assessment section
    content.append("\n")
    content.append("Risk: ", style="bold")
    content.append_text(_risk_badge(combined_level))
    content.append(f"(score: {risk_assessment.score}/100)\n", style="dim")

    # Risk reasons
    all_reasons = list(dict.fromkeys(
        risk_assessment.reasons + [explainer_result.risk_reason]
    ))
    if all_reasons:
        content.append("\nRisk factors:\n", style="bold")
        for reason in all_reasons:
            if reason:
                content.append(f"  • {escape(reason)}\n", style="safe_run.label")

    # Effects
    if explainer_result.effects:
        content.append("\nEffects:\n", style="bold")
        for effect in explainer_result.effects:
            content.append(f"  • {escape(effect)}\n", style="dim")

    # Reversibility
    reversible_text = "Yes" if explainer_result.reversible else "No"
    reversible_style = "green" if explainer_result.reversible else "red"
    content.append("\nReversible: ", style="bold")
    content.append(reversible_text, style=reversible_style)
    content.append("\n")

    # Provider info
    if explainer_result.is_fallback:
        content.append(
            "\n⚠  LLM unavailable — explanation is a fallback.\n",
            style="safe_run.warning",
        )
    else:
        content.append(
            f"\nExplained by: {explainer_result.provider_used}/{explainer_result.model_used}",
            style="safe_run.provider",
        )

    title = _build_panel_title(combined_level)

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def _build_panel_title(level: RiskLevel) -> Text:
    """Build the Rich Text title for the explanation panel.

    Args:
        level: The combined risk level.

    Returns:
        A styled Text object for the panel title.
    """
    emoji = get_risk_level_emoji(level)
    color = get_risk_level_color(level)
    text = Text()
    text.append(" safe_run ", style="bold")
    text.append(f"{emoji} ", style=color)
    text.append(level.value, style=f"bold {color}")
    text.append(" ")
    return text


# ---------------------------------------------------------------------------
# Risk-only display (for bypass mode and quick status)
# ---------------------------------------------------------------------------


def display_risk_badge(
    risk_assessment: RiskAssessment,
    *,
    console: Console = console,
) -> None:
    """Render a compact risk badge line to the terminal.

    Useful for quick status display without a full explanation panel.

    Args:
        risk_assessment: The result from the rule-based risk pre-screener.
        console: Rich Console to write to.
    """
    badge = _risk_badge(risk_assessment.level)
    color = get_risk_level_color(risk_assessment.level)
    line = Text()
    line.append("[safe_run] Risk: ", style="bold")
    line.append_text(badge)
    line.append(f" score={risk_assessment.score}/100", style="dim")
    console.print(line)


# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------


def prompt_confirmation(
    risk_level: RiskLevel,
    command: str = "",
    *,
    auto_confirm: bool = False,
    console: Console = console,
) -> ConfirmationChoice:
    """Display a confirmation prompt and return the user's choice.

    Behaviour varies by risk level:

    - **LOW / MEDIUM**: If ``auto_confirm=True``, the command is approved
      automatically. Otherwise the user is prompted with a simple yes/no.
    - **HIGH**: Always prompts the user, even if ``auto_confirm=True``.
    - **CRITICAL**: Requires the user to type ``"yes"`` (exact, lowercase)
      to confirm. A simple ``y`` or ``Enter`` is rejected.

    Args:
        risk_level: The combined risk level of the command.
        command: The command string (used in the prompt message).
        auto_confirm: If True, auto-approve commands below the HIGH threshold
            without asking.
        console: Rich Console to write to.

    Returns:
        A ConfirmationChoice enum value.
    """
    # Auto-confirm for LOW/MEDIUM when configured
    if auto_confirm and risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        _print_auto_confirmed(risk_level, console=console)
        return ConfirmationChoice.YES

    # CRITICAL: require explicit typed confirmation
    if risk_level == RiskLevel.CRITICAL:
        return _prompt_critical_confirmation(command, console=console)

    # HIGH + LOW/MEDIUM without auto_confirm: standard yes/no prompt
    return _prompt_standard_confirmation(risk_level, command, console=console)


def _print_auto_confirmed(
    risk_level: RiskLevel,
    *,
    console: Console,
) -> None:
    """Print an auto-confirmation notice to the console.

    Args:
        risk_level: The risk level that was auto-confirmed.
        console: Rich Console to write to.
    """
    badge = _risk_badge(risk_level)
    line = Text()
    line.append("✓ Auto-confirmed ", style="green")
    line.append_text(badge)
    line.append(" command.", style="dim")
    console.print(line)


def _prompt_standard_confirmation(
    risk_level: RiskLevel,
    command: str,
    *,
    console: Console,
) -> ConfirmationChoice:
    """Render a standard yes/no confirmation prompt.

    Args:
        risk_level: The risk level of the command.
        command: The command string.
        console: Rich Console to write to.

    Returns:
        ConfirmationChoice.YES, ConfirmationChoice.NO, or
        ConfirmationChoice.ABORT on keyboard interrupt.
    """
    color = get_risk_level_color(risk_level)
    badge = _risk_badge(risk_level)

    prompt_text = Text()
    prompt_text.append("\nRun this ")
    prompt_text.append_text(badge)
    prompt_text.append(" command? ", style="bold")
    prompt_text.append("[y/N]", style="dim")

    console.print(prompt_text, end=" ")

    try:
        response = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold red]Aborted.[/bold red]")
        return ConfirmationChoice.ABORT

    if response in ("y", "yes"):
        return ConfirmationChoice.YES
    else:
        _print_declined(console=console)
        return ConfirmationChoice.NO


def _prompt_critical_confirmation(
    command: str,
    *,
    console: Console,
) -> ConfirmationChoice:
    """Render a high-friction confirmation prompt for CRITICAL commands.

    The user must type the exact string ``yes`` (lowercase, no variations)
    to proceed. Any other input declines.

    Args:
        command: The command string (shown in the warning).
        console: Rich Console to write to.

    Returns:
        ConfirmationChoice.YES, ConfirmationChoice.NO, or
        ConfirmationChoice.ABORT on keyboard interrupt.
    """
    console.print()
    warning = Panel(
        Text.from_markup(
            "[bold red]⚠  CRITICAL RISK COMMAND[/bold red]\n\n"
            "This command may cause [bold]irreversible[/bold] damage to your system,\n"
            "data, or security. It [bold]cannot be undone[/bold].\n\n"
            "[dim]If you are absolutely certain, type [bold]yes[/bold] "
            "(exactly, lowercase) to continue.\n"
            "Any other input will abort.[/dim]"
        ),
        border_style="bold red",
        box=box.HEAVY,
        padding=(1, 2),
        title=" 💀 CRITICAL ",
        title_align="center",
    )
    console.print(warning)
    console.print()

    prompt_line = Text()
    prompt_line.append("Type ", style="bold")
    prompt_line.append("yes", style="bold red")
    prompt_line.append(" to confirm, anything else to abort: ", style="bold")
    console.print(prompt_line, end="")

    try:
        response = input().strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold red]Aborted.[/bold red]")
        return ConfirmationChoice.ABORT

    if response == "yes":
        return ConfirmationChoice.YES
    else:
        _print_declined(console=console)
        return ConfirmationChoice.NO


def _print_declined(*, console: Console) -> None:
    """Print a 'command declined' message.

    Args:
        console: Rich Console to write to.
    """
    console.print("\n[bold yellow]Command not executed.[/bold yellow]")


# ---------------------------------------------------------------------------
# Blocked command display
# ---------------------------------------------------------------------------


def display_blocked(
    command: str,
    reason: str = "",
    *,
    console: Console = console,
) -> None:
    """Display a message indicating the command was blocked by the blocklist.

    Args:
        command: The command that was blocked.
        reason: Optional explanation of why it was blocked.
        console: Rich Console to write to.
    """
    content = Text()
    content.append("🚫  This command is blocked by your configuration.\n\n", style="bold red")
    content.append("Command: ", style="bold")
    content.append(escape(command), style="safe_run.command")
    if reason:
        content.append("\n\nReason: ", style="bold")
        content.append(escape(reason), style="dim")

    panel = Panel(
        content,
        title=" Blocked Command ",
        title_align="left",
        border_style="red",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


# ---------------------------------------------------------------------------
# Allowlisted command display
# ---------------------------------------------------------------------------


def display_allowlisted(
    command: str,
    *,
    console: Console = console,
) -> None:
    """Display a brief message indicating the command is on the allowlist.

    Args:
        command: The command that was allowlisted.
        console: Rich Console to write to.
    """
    line = Text()
    line.append("✓ Allowlisted: ", style="bold green")
    line.append(escape(command), style="safe_run.command")
    console.print(line)


# ---------------------------------------------------------------------------
# Execution result display
# ---------------------------------------------------------------------------


def display_execution_result(
    result: ExecutionResult,
    *,
    verbose: bool = False,
    console: Console = console,
) -> None:
    """Display a summary of a completed command execution.

    Shows exit code, duration, and (if ``verbose=True`` or the command
    failed) any captured stdout/stderr.

    Args:
        result: The ExecutionResult from executor.execute_command.
        verbose: If True, always show stdout/stderr even on success.
        console: Rich Console to write to.
    """
    console.print()
    console.print(Rule(style="dim"))

    # Status line
    status_text = Text()
    if result.interrupted:
        status_text.append("⚡ Interrupted", style="bold yellow")
    elif result.timed_out:
        status_text.append("⏱  Timed out", style="bold red")
    elif result.succeeded:
        status_text.append("✓ ", style="bold green")
        status_text.append("Success", style="bold green")
    else:
        status_text.append("✗ ", style="bold red")
        status_text.append("Failed", style="bold red")

    status_text.append(f"  exit={result.exit_code}", style="dim")
    status_text.append(
        f"  duration={result.duration_seconds:.2f}s", style="dim"
    )
    console.print(status_text)

    # Show captured output if verbose or on failure
    show_output = verbose or result.failed or result.timed_out

    if show_output and result.stdout.strip():
        console.print("\n[bold]stdout:[/bold]")
        console.print(result.stdout, style="dim")

    if show_output and result.stderr.strip():
        console.print("\n[bold]stderr:[/bold]")
        console.print(result.stderr, style="dim yellow")


# ---------------------------------------------------------------------------
# Error display helpers
# ---------------------------------------------------------------------------


def display_error(
    message: str,
    *,
    title: str = "Error",
    console: Console = console,
) -> None:
    """Display a generic error message in a red panel.

    Args:
        message: The error message to display.
        title: Panel title. Defaults to 'Error'.
        console: Rich Console to write to.
    """
    panel = Panel(
        Text(escape(message), style="red"),
        title=f" {title} ",
        title_align="left",
        border_style="red",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    console.print()
    console.print(panel)


def display_warning(
    message: str,
    *,
    title: str = "Warning",
    console: Console = console,
) -> None:
    """Display a warning message in a yellow panel.

    Args:
        message: The warning message to display.
        title: Panel title. Defaults to 'Warning'.
        console: Rich Console to write to.
    """
    panel = Panel(
        Text(escape(message), style="yellow"),
        title=f" ⚠  {title} ",
        title_align="left",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    console.print()
    console.print(panel)


def display_info(
    message: str,
    *,
    title: str = "Info",
    console: Console = console,
) -> None:
    """Display an informational message in a blue panel.

    Args:
        message: The informational message to display.
        title: Panel title. Defaults to 'Info'.
        console: Rich Console to write to.
    """
    panel = Panel(
        Text(escape(message), style="blue"),
        title=f" ℹ  {title} ",
        title_align="left",
        border_style="blue",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    console.print()
    console.print(panel)


# ---------------------------------------------------------------------------
# Bypass mode notification
# ---------------------------------------------------------------------------


def display_bypass_notice(
    command: str,
    *,
    console: Console = console,
) -> None:
    """Display a brief notice that safe_run is operating in bypass mode.

    In bypass mode the command is executed without explanation or confirmation.

    Args:
        command: The command being bypassed.
        console: Rich Console to write to.
    """
    line = Text()
    line.append("[safe_run] ", style="dim")
    line.append("Bypass mode", style="bold yellow")
    line.append(" — executing without explanation: ", style="dim")
    line.append(escape(command), style="safe_run.command")
    console.print(line)


# ---------------------------------------------------------------------------
# Loading / spinner display
# ---------------------------------------------------------------------------


def display_loading(
    message: str = "Consulting LLM...",
    *,
    console: Console = console,
) -> None:
    """Print a simple 'loading' status line.

    This is a non-blocking status indicator for use before a blocking LLM
    call. For animated spinners, callers should use Rich's ``Status`` context
    manager directly.

    Args:
        message: The status message to display.
        console: Rich Console to write to.
    """
    line = Text()
    line.append("⟳ ", style="bold blue")
    line.append(message, style="dim")
    console.print(line)


# ---------------------------------------------------------------------------
# Compact summary table (for --list-recent style views)
# ---------------------------------------------------------------------------


def build_risk_summary_table(
    entries: list[tuple[str, RiskLevel, int]],
) -> Table:
    """Build a Rich Table summarising multiple command risk assessments.

    Args:
        entries: A list of (command, risk_level, score) tuples.

    Returns:
        A Rich Table ready to be printed.
    """
    table = Table(
        title="Command Risk Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Command", style="safe_run.command", no_wrap=False)
    table.add_column("Risk Level", justify="center", width=12)
    table.add_column("Score", justify="right", width=7)

    for idx, (cmd, level, score) in enumerate(entries, start=1):
        color = get_risk_level_color(level)
        emoji = get_risk_level_emoji(level)
        risk_cell = Text(f"{emoji} {level.value}", style=f"bold {color}")
        table.add_row(
            str(idx),
            escape(cmd[:80]) + ("…" if len(cmd) > 80 else ""),
            risk_cell,
            str(score),
        )

    return table
