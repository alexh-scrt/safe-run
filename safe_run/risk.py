"""Rule-based risk pre-screener for safe_run.

This module implements a pattern-matching engine that detects dangerous shell
command patterns and assigns them a risk level (LOW, MEDIUM, HIGH, or CRITICAL)
before any LLM call is made. It covers 30+ dangerous patterns including file
deletions, permission changes, network operations, privilege escalation,
disk operations, and more.

Typical usage example::

    from safe_run.risk import assess_risk, RiskLevel, RiskAssessment

    result = assess_risk("rm -rf /")
    print(result.level)       # RiskLevel.CRITICAL
    print(result.reasons)     # ['Recursive forced deletion from root or home']
    print(result.score)       # 100
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, NamedTuple, Optional, Pattern, Sequence


class RiskLevel(str, Enum):
    """Enumeration of risk severity levels in ascending order.

    Attributes:
        LOW: Minimal risk; informational commands.
        MEDIUM: Moderate risk; operations that may have side effects.
        HIGH: Significant risk; operations that can cause data loss or
            security issues if misused.
        CRITICAL: Extreme risk; operations that can cause irreversible
            system damage, root-level access, or full data loss.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return _LEVEL_ORDER.index(self) < _LEVEL_ORDER.index(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return _LEVEL_ORDER.index(self) <= _LEVEL_ORDER.index(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return _LEVEL_ORDER.index(self) > _LEVEL_ORDER.index(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return _LEVEL_ORDER.index(self) >= _LEVEL_ORDER.index(other)


# Ordered list of levels from lowest to highest severity
_LEVEL_ORDER: List[RiskLevel] = [
    RiskLevel.LOW,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
]

# Map risk level to numeric score for aggregation
_LEVEL_SCORE = {
    RiskLevel.LOW: 10,
    RiskLevel.MEDIUM: 30,
    RiskLevel.HIGH: 60,
    RiskLevel.CRITICAL: 100,
}


@dataclass(frozen=True)
class RiskAssessment:
    """The result of a risk assessment for a single shell command.

    Attributes:
        level: The overall risk level assigned to the command.
        score: A numeric score (0-100) representing aggregate risk severity.
        reasons: Human-readable descriptions of each matched risk pattern.
        matched_rules: Names of the risk rules that fired.
        command: The original command string that was assessed.
    """

    level: RiskLevel
    score: int
    reasons: List[str]
    matched_rules: List[str]
    command: str

    @property
    def is_dangerous(self) -> bool:
        """Return True if the risk level is HIGH or CRITICAL.

        Returns:
            True for HIGH or CRITICAL risk levels.
        """
        return self.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def requires_confirmation(self) -> bool:
        """Return True if the command should always require user confirmation.

        CRITICAL commands must always be confirmed regardless of config
        threshold settings.

        Returns:
            True for CRITICAL risk level.
        """
        return self.level == RiskLevel.CRITICAL


class RiskRule(NamedTuple):
    """A single pattern-based risk detection rule.

    Attributes:
        name: Unique identifier for this rule.
        pattern: Compiled regular expression to match against the command.
        level: Risk level assigned when this rule matches.
        reason: Human-readable explanation of why this is risky.
    """

    name: str
    pattern: Pattern[str]
    level: RiskLevel
    reason: str


def _rule(
    name: str,
    regex: str,
    level: RiskLevel,
    reason: str,
    flags: int = re.IGNORECASE,
) -> RiskRule:
    """Convenience factory to create a RiskRule with a compiled pattern.

    Args:
        name: Unique rule identifier.
        regex: Regular expression string.
        level: Risk level for matches.
        reason: Human-readable match description.
        flags: re flags (default: re.IGNORECASE).

    Returns:
        A compiled RiskRule instance.
    """
    return RiskRule(
        name=name,
        pattern=re.compile(regex, flags),
        level=level,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Rule definitions — 30+ patterns covering the most dangerous shell operations
# ---------------------------------------------------------------------------

RISK_RULES: List[RiskRule] = [
    # ------------------------------------------------------------------ CRITICAL
    _rule(
        name="rm_rf_root",
        regex=r"\brm\b.*\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*$|\brm\b.*\s+-[a-z]*f[a-z]*r[a-z]*\s+/\s*$",
        level=RiskLevel.CRITICAL,
        reason="Recursive forced deletion from filesystem root — will destroy the entire OS",
    ),
    _rule(
        name="rm_rf_home",
        regex=r"\brm\b.*-[a-z]*r[a-z]*f[a-z]*.*[\$~](?:HOME)?\b|\brm\b.*-[a-z]*f[a-z]*r[a-z]*.*[\$~](?:HOME)?\b",
        level=RiskLevel.CRITICAL,
        reason="Recursive forced deletion of home directory — will destroy all user data",
    ),
    _rule(
        name="rm_rf_general",
        regex=r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)",
        level=RiskLevel.HIGH,
        reason="Recursive forced file/directory deletion (rm -rf)",
    ),
    _rule(
        name="curl_pipe_shell",
        regex=r"\bcurl\b[^|]*\|\s*(?:ba)?sh\b|\bcurl\b[^|]*\|\s*python[23]?\b|\bcurl\b[^|]*\|\s*perl\b|\bcurl\b[^|]*\|\s*ruby\b",
        level=RiskLevel.CRITICAL,
        reason="Downloads and pipes remote code directly into a shell interpreter — arbitrary code execution risk",
    ),
    _rule(
        name="wget_pipe_shell",
        regex=r"\bwget\b[^|]*\|\s*(?:ba)?sh\b|\bwget\b[^|]*\|\s*python[23]?\b|\bwget\b[^|]*\|\s*perl\b",
        level=RiskLevel.CRITICAL,
        reason="Downloads and pipes remote content directly into a shell interpreter — arbitrary code execution risk",
    ),
    _rule(
        name="dd_of_disk",
        regex=r"\bdd\b.*\bof=\s*/dev/(?:sd[a-z]|hd[a-z]|nvme\d|sda|disk\d|vda)",
        level=RiskLevel.CRITICAL,
        reason="Writes directly to a raw disk device — will overwrite partition table or disk data",
    ),
    _rule(
        name="mkfs",
        regex=r"\bmkfs(?:\.[a-z0-9]+)?\s+/dev/",
        level=RiskLevel.CRITICAL,
        reason="Formats a disk or partition — all data on the target will be erased",
    ),
    _rule(
        name="fork_bomb",
        regex=r":\(\)\s*\{\s*:|&\s*\}\s*;\s*:",
        level=RiskLevel.CRITICAL,
        reason="Fork bomb detected — will exhaust system resources and cause a crash",
    ),
    _rule(
        name="overwrite_etc_passwd",
        regex=r">\s*/etc/(?:passwd|shadow|sudoers|hosts|fstab|crontab)",
        level=RiskLevel.CRITICAL,
        reason="Overwrites a critical system file — can render the system unbootable or insecure",
    ),
    _rule(
        name="python_exec_eval_network",
        regex=r"python[23]?\s+-c\s+['\"].*(?:exec|eval|__import__).*(?:urllib|http|socket)",
        level=RiskLevel.CRITICAL,
        reason="Python one-liner downloading and executing remote code",
    ),
    _rule(
        name="base64_decode_pipe_shell",
        regex=r"\bbase64\b.*\|\s*(?:ba)?sh\b|\bbase64\b.*\|\s*python[23]?\b",
        level=RiskLevel.CRITICAL,
        reason="Decodes base64 content and pipes to a shell — common obfuscated code execution technique",
    ),
    _rule(
        name="history_wipe",
        regex=r"\bhistory\s+-[cw]\b|>\s*~?/?\.(?:bash|zsh)_history\b|\brm\b.*\.(?:bash|zsh)_history",
        level=RiskLevel.HIGH,
        reason="Clears or overwrites shell history — common anti-forensics technique",
    ),
    # ------------------------------------------------------------------ HIGH
    _rule(
        name="chmod_777",
        regex=r"\bchmod\b.*\b777\b",
        level=RiskLevel.HIGH,
        reason="Sets world-readable/writable/executable permissions (chmod 777) — severe security risk",
    ),
    _rule(
        name="chmod_recursive_permissive",
        regex=r"\bchmod\b.*-[a-zA-Z]*R[a-zA-Z]*.*(?:777|a\+(?:rwx|x)|o\+(?:rw|w))",
        level=RiskLevel.HIGH,
        reason="Recursively sets permissive permissions — affects all files in directory tree",
    ),
    _rule(
        name="chown_root",
        regex=r"\bchown\b.*\broot\b",
        level=RiskLevel.HIGH,
        reason="Changes file ownership to root — can escalate privileges if combined with setuid",
    ),
    _rule(
        name="sudo_command",
        regex=r"\bsudo\b",
        level=RiskLevel.HIGH,
        reason="Executes command with superuser (root) privileges",
    ),
    _rule(
        name="su_root",
        regex=r"\bsu\b(?:\s+-[a-z])*\s+(?:root|-)",
        level=RiskLevel.HIGH,
        reason="Switches to root user account",
    ),
    _rule(
        name="nohup_background",
        regex=r"\bnohup\b.*&\s*$",
        level=RiskLevel.MEDIUM,
        reason="Runs process in background immune to hangup — may persist after session ends",
    ),
    _rule(
        name="iptables_flush",
        regex=r"\biptables\b.*-[FXZ]\b|\biptables\b.*--flush\b",
        level=RiskLevel.HIGH,
        reason="Flushes firewall rules — removes all network security protections",
    ),
    _rule(
        name="iptables_allow_all",
        regex=r"\biptables\b.*-P\s+(?:INPUT|OUTPUT|FORWARD)\s+ACCEPT",
        level=RiskLevel.HIGH,
        reason="Sets default firewall policy to ACCEPT — opens all ports to network traffic",
    ),
    _rule(
        name="ufw_disable",
        regex=r"\bufw\s+disable\b",
        level=RiskLevel.HIGH,
        reason="Disables the Uncomplicated Firewall — removes network protection",
    ),
    _rule(
        name="systemctl_disable",
        regex=r"\bsystemctl\b.*\bdisable\b",
        level=RiskLevel.MEDIUM,
        reason="Disables a system service — may affect system functionality on next boot",
    ),
    _rule(
        name="systemctl_stop_critical",
        regex=r"\bsystemctl\b.*\bstop\b.*(?:ssh|networking|network|firewall|fail2ban|ufw)",
        level=RiskLevel.HIGH,
        reason="Stops a critical system service — may cause connectivity loss or security exposure",
    ),
    _rule(
        name="kill_all",
        regex=r"\bkillall\b|\bpkill\b.*-9|\bkill\b.*-9\s+-1\b",
        level=RiskLevel.HIGH,
        reason="Forcefully terminates multiple processes — may cause data loss or system instability",
    ),
    _rule(
        name="shutdown_reboot",
        regex=r"\b(?:shutdown|reboot|halt|poweroff|init\s+[06])\b",
        level=RiskLevel.HIGH,
        reason="Shuts down or reboots the system — will terminate all running processes",
    ),
    _rule(
        name="write_cron",
        regex=r"(?:crontab\s+-[re]|>\s*/etc/cron\.|echo.*>.*crontab|tee.*cron)",
        level=RiskLevel.HIGH,
        reason="Modifies cron jobs — can schedule persistent arbitrary code execution",
    ),
    _rule(
        name="suid_bit",
        regex=r"\bchmod\b.*[ug]\+s\b|\bchmod\b.*(?:4[0-9]{3}|2[0-9]{3})\b",
        level=RiskLevel.HIGH,
        reason="Sets setuid/setgid bit — executable will run with owner's privileges, a common privilege escalation vector",
    ),
    _rule(
        name="write_ssh_authorized_keys",
        regex=r"(?:>>|>)\s*~?/?(?:\$HOME/)?(?:\.ssh/authorized_keys|/root/\.ssh/authorized_keys)",
        level=RiskLevel.HIGH,
        reason="Writes to SSH authorized_keys — grants persistent remote access without a password",
    ),
    _rule(
        name="nc_reverse_shell",
        regex=r"\bnc\b.*-[a-zA-Z]*e[a-zA-Z]*.*(?:/bin/(?:ba)?sh|cmd)|\bncat\b.*--exec",
        level=RiskLevel.CRITICAL,
        reason="Netcat reverse shell — establishes covert remote shell access",
    ),
    _rule(
        name="python_reverse_shell",
        regex=r"python[23]?.*socket.*(?:connect|SOCK_STREAM).*(?:exec|dup2|sh)",
        level=RiskLevel.CRITICAL,
        reason="Python reverse shell pattern — establishes covert remote access",
    ),
    _rule(
        name="truncate_disk",
        regex=r">\s*/dev/(?:sd[a-z]|hd[a-z]|nvme\d|vda)",
        level=RiskLevel.CRITICAL,
        reason="Truncates a raw disk device — destroys all data on the disk",
    ),
    _rule(
        name="rm_system_dirs",
        regex=r"\brm\b.*-[a-zA-Z]*r[a-zA-Z]*.*\s+(?:/usr|/lib|/bin|/sbin|/boot|/etc|/var)\b",
        level=RiskLevel.CRITICAL,
        reason="Recursive deletion of a critical system directory — will render the system inoperable",
    ),
    _rule(
        name="wget_execute",
        regex=r"\bwget\b.*-O\s*-\s*.*\|\s*(?:ba)?sh\b|\bwget\b.*(?:--output-document=-).*\|\s*(?:ba)?sh\b",
        level=RiskLevel.CRITICAL,
        reason="Downloads remote script and pipes to shell — arbitrary remote code execution",
    ),
    # ------------------------------------------------------------------ MEDIUM
    _rule(
        name="rm_force",
        regex=r"\brm\b.*\s+-[a-zA-Z]*f[a-zA-Z]*(?!r)",
        level=RiskLevel.MEDIUM,
        reason="Forced file deletion (rm -f) — skips confirmation and deletes without warning",
    ),
    _rule(
        name="curl_insecure",
        regex=r"\bcurl\b.*(?:-k|--insecure)\b",
        level=RiskLevel.MEDIUM,
        reason="Disables SSL certificate verification — vulnerable to MITM attacks",
    ),
    _rule(
        name="wget_no_check_certificate",
        regex=r"\bwget\b.*--no-check-certificate\b",
        level=RiskLevel.MEDIUM,
        reason="Disables SSL certificate verification for wget — vulnerable to MITM attacks",
    ),
    _rule(
        name="chmod_recursive",
        regex=r"\bchmod\b.*-[a-zA-Z]*R[a-zA-Z]*",
        level=RiskLevel.MEDIUM,
        reason="Recursively changes permissions on a directory tree — can affect many files",
    ),
    _rule(
        name="chown_recursive",
        regex=r"\bchown\b.*-[a-zA-Z]*R[a-zA-Z]*",
        level=RiskLevel.MEDIUM,
        reason="Recursively changes ownership of a directory tree",
    ),
    _rule(
        name="overwrite_redirect",
        regex=r"\w+.*>(?!>)\s*(?!/dev/null)\S+",
        level=RiskLevel.LOW,
        reason="Output redirection overwrites a file — existing contents will be lost",
    ),
    _rule(
        name="git_force_push",
        regex=r"\bgit\b.*\bpush\b.*(?:-f|--force)",
        level=RiskLevel.MEDIUM,
        reason="Force push to git remote — rewrites remote history and may cause data loss for collaborators",
    ),
    _rule(
        name="git_reset_hard",
        regex=r"\bgit\b.*\breset\b.*--hard",
        level=RiskLevel.MEDIUM,
        reason="Hard git reset — discards all uncommitted changes permanently",
    ),
    _rule(
        name="git_clean_force",
        regex=r"\bgit\b.*\bclean\b.*-[a-zA-Z]*f[a-zA-Z]*",
        level=RiskLevel.MEDIUM,
        reason="Forcefully removes untracked files from git working tree — cannot be undone",
    ),
    _rule(
        name="drop_database",
        regex=r"\bDROP\s+(?:DATABASE|TABLE|SCHEMA)\b",
        level=RiskLevel.HIGH,
        reason="SQL DROP statement — permanently deletes a database, table, or schema and all its data",
    ),
    _rule(
        name="mysql_dump_credentials",
        regex=r"\bmysql(?:dump)?\b.*-p\S+|\bpsql\b.*password=\S+",
        level=RiskLevel.MEDIUM,
        reason="Database credentials visible in command line — may be captured in shell history or process list",
    ),
    _rule(
        name="env_credentials",
        regex=r"(?:PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY)=[^\s]+",
        level=RiskLevel.MEDIUM,
        reason="Sensitive credentials set via environment variable in command — visible in process list and history",
    ),
    _rule(
        name="network_scan",
        regex=r"\bnmap\b|\bmasscan\b|\bzmap\b",
        level=RiskLevel.MEDIUM,
        reason="Network scanning tool — may violate network policies or terms of service",
    ),
    _rule(
        name="ssh_no_host_check",
        regex=r"\bssh\b.*-o\s*StrictHostKeyChecking=no",
        level=RiskLevel.MEDIUM,
        reason="Disables SSH host key verification — vulnerable to MITM attacks",
    ),
    _rule(
        name="xargs_rm",
        regex=r"\bxargs\b.*\brm\b",
        level=RiskLevel.MEDIUM,
        reason="Passes piped input to rm — can delete many files based on dynamic input",
    ),
    _rule(
        name="find_delete",
        regex=r"\bfind\b.*(?:-delete|-exec\s+rm)",
        level=RiskLevel.MEDIUM,
        reason="find with deletion action — can delete many files matching a pattern",
    ),
    _rule(
        name="pipe_to_sudo",
        regex=r"\|\s*sudo\b",
        level=RiskLevel.HIGH,
        reason="Pipes command output into sudo — executes potentially untrusted content with root privileges",
    ),
    _rule(
        name="eval_command",
        regex=r"\beval\s+",
        level=RiskLevel.MEDIUM,
        reason="eval executes a dynamically constructed string as a shell command — code injection risk",
    ),
    _rule(
        name="scp_recursive",
        regex=r"\bscp\b.*-[a-zA-Z]*r[a-zA-Z]*",
        level=RiskLevel.LOW,
        reason="Recursively copies files over SSH — transfers potentially large amounts of data",
    ),
    _rule(
        name="rsync_delete",
        regex=r"\brsync\b.*--delete",
        level=RiskLevel.MEDIUM,
        reason="rsync with --delete removes files from destination that are not in source — can cause data loss",
    ),
    _rule(
        name="mount_command",
        regex=r"\bmount\b\s+(?!-l|--list|--show)",
        level=RiskLevel.MEDIUM,
        reason="Mounts a filesystem — can expose or modify disk contents",
    ),
    _rule(
        name="umount_command",
        regex=r"\bumount\b",
        level=RiskLevel.MEDIUM,
        reason="Unmounts a filesystem — may interrupt running processes that use it",
    ),
    _rule(
        name="passwd_change",
        regex=r"\bpasswd\b(?:\s+\S+)?",
        level=RiskLevel.MEDIUM,
        reason="Changes a user account password",
    ),
    _rule(
        name="useradd_mod",
        regex=r"\b(?:useradd|usermod|userdel|groupadd|groupmod|groupdel)\b",
        level=RiskLevel.MEDIUM,
        reason="Modifies user or group accounts — affects system access control",
    ),
    _rule(
        name="visudo_write",
        regex=r"\bvisudo\b|(?:tee|>>|>)\s*/etc/sudoers",
        level=RiskLevel.HIGH,
        reason="Modifies sudoers configuration — controls who can run commands as root",
    ),
    # ------------------------------------------------------------------ LOW
    _rule(
        name="curl_download",
        regex=r"\bcurl\b.*(?:-O|-o\s+\S+|--output)\b",
        level=RiskLevel.LOW,
        reason="Downloads a file from a remote URL",
    ),
    _rule(
        name="wget_download",
        regex=r"\bwget\b\s+(?!-[a-zA-Z]*O\s*-\s)\S+",
        level=RiskLevel.LOW,
        reason="Downloads a file from a remote URL",
    ),
    _rule(
        name="pip_install",
        regex=r"\bpip[23]?\b.*\binstall\b",
        level=RiskLevel.LOW,
        reason="Installs Python packages — packages may contain arbitrary code that runs on install",
    ),
    _rule(
        name="npm_install",
        regex=r"\bnpm\b.*\binstall\b|\byarn\b.*\badd\b",
        level=RiskLevel.LOW,
        reason="Installs Node.js packages — packages may contain arbitrary code that runs on install",
    ),
    _rule(
        name="apt_install",
        regex=r"\b(?:apt|apt-get|yum|dnf|zypper|pacman)\b.*\b(?:install|remove|purge)\b",
        level=RiskLevel.LOW,
        reason="Installs or removes system packages — changes system software configuration",
    ),
]


def _normalise_command(command: str) -> str:
    """Normalise a shell command string for pattern matching.

    Collapses multiple whitespace characters into single spaces and strips
    leading/trailing whitespace. Does not modify quoting or escaping.

    Args:
        command: The raw shell command string.

    Returns:
        A normalised copy of the command.
    """
    return re.sub(r"\s+", " ", command).strip()


def _compute_aggregate_score(matched_levels: List[RiskLevel]) -> int:
    """Compute an aggregate risk score from a list of matched risk levels.

    Takes the maximum score and adds partial credit for additional matches,
    capped at 100.

    Args:
        matched_levels: List of RiskLevel values from matched rules.

    Returns:
        An integer score in the range [0, 100].
    """
    if not matched_levels:
        return 0

    scores = [_LEVEL_SCORE[lvl] for lvl in matched_levels]
    primary = max(scores)
    bonus = sum(s // 5 for s in scores if s != primary)
    return min(100, primary + bonus)


def _dominant_level(matched_levels: List[RiskLevel]) -> RiskLevel:
    """Return the highest risk level from a list of matched levels.

    Args:
        matched_levels: Non-empty list of RiskLevel values.

    Returns:
        The maximum RiskLevel.
    """
    return max(matched_levels, key=lambda lvl: _LEVEL_ORDER.index(lvl))


def assess_risk(
    command: str,
    rules: Optional[Sequence[RiskRule]] = None,
) -> RiskAssessment:
    """Assess the risk of a shell command against the rule set.

    Matches the command string against all rules (or a provided custom list)
    and returns a RiskAssessment summarising the overall risk level, a numeric
    score, human-readable reasons, and the names of matched rules.

    If no rules match, the command is assigned RiskLevel.LOW with a score of 0.

    Args:
        command: The shell command string to evaluate.
        rules: Optional list of RiskRule objects to use. If None, the default
            RISK_RULES list is used.

    Returns:
        A RiskAssessment for the given command.
    """
    if rules is None:
        rules = RISK_RULES

    normalised = _normalise_command(command)

    matched_levels: List[RiskLevel] = []
    reasons: List[str] = []
    matched_rule_names: List[str] = []

    seen_reasons: set[str] = set()

    for rule in rules:
        if rule.pattern.search(normalised):
            matched_levels.append(rule.level)
            matched_rule_names.append(rule.name)
            if rule.reason not in seen_reasons:
                reasons.append(rule.reason)
                seen_reasons.add(rule.reason)

    if not matched_levels:
        return RiskAssessment(
            level=RiskLevel.LOW,
            score=0,
            reasons=[],
            matched_rules=[],
            command=command,
        )

    level = _dominant_level(matched_levels)
    score = _compute_aggregate_score(matched_levels)

    return RiskAssessment(
        level=level,
        score=score,
        reasons=reasons,
        matched_rules=matched_rule_names,
        command=command,
    )


def get_risk_level_color(level: RiskLevel) -> str:
    """Return the Rich markup color string for a given risk level.

    Args:
        level: A RiskLevel enum value.

    Returns:
        A Rich color name suitable for use in markup, e.g. 'green'.
    """
    colors = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "orange3",
        RiskLevel.CRITICAL: "bold red",
    }
    return colors.get(level, "white")


def get_risk_level_emoji(level: RiskLevel) -> str:
    """Return an emoji string representing a given risk level.

    Args:
        level: A RiskLevel enum value.

    Returns:
        An emoji string.
    """
    emojis = {
        RiskLevel.LOW: "\u2705",       # ✅
        RiskLevel.MEDIUM: "\u26a0\ufe0f",  # ⚠️
        RiskLevel.HIGH: "\U0001f6a8",   # 🚨
        RiskLevel.CRITICAL: "\U0001f480",  # 💀
    }
    return emojis.get(level, "❓")
