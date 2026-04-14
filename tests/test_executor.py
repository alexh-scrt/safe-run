"""Unit tests for safe_run.executor module.

Covers command execution, timeout handling, output capture, streaming,
exit code handling, and error conditions.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from safe_run.executor import (
    CommandNotFoundError,
    CommandTimeoutError,
    ExecutionResult,
    ExecutorError,
    _decode_output,
    _terminate_process,
    execute_command,
    execute_command_with_output_callback,
)


# ---------------------------------------------------------------------------
# ExecutionResult properties
# ---------------------------------------------------------------------------


class TestExecutionResultProperties:
    def test_succeeded_on_zero_exit(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.succeeded is True

    def test_failed_on_nonzero_exit(self) -> None:
        result = ExecutionResult(command="ls", exit_code=1)
        assert result.failed is True

    def test_succeeded_false_on_nonzero(self) -> None:
        result = ExecutionResult(command="ls", exit_code=2)
        assert result.succeeded is False

    def test_failed_false_on_zero(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.failed is False

    def test_default_stdout_empty(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.stdout == ""

    def test_default_stderr_empty(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.stderr == ""

    def test_default_timed_out_false(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.timed_out is False

    def test_default_interrupted_false(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.interrupted is False

    def test_timed_out_attribute(self) -> None:
        result = ExecutionResult(command="sleep 10", exit_code=-1, timed_out=True)
        assert result.timed_out is True

    def test_duration_stored(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0, duration_seconds=1.5)
        assert result.duration_seconds == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# _decode_output
# ---------------------------------------------------------------------------


class TestDecodeOutput:
    def test_decodes_utf8(self) -> None:
        raw = b"hello world\n"
        assert _decode_output(raw, 1024) == "hello world\n"

    def test_decodes_latin1_fallback(self) -> None:
        raw = bytes([0xFF, 0xFE, 0x41])  # Invalid UTF-8 sequence
        result = _decode_output(raw, 1024)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncation_adds_notice(self) -> None:
        raw = b"x" * 100
        result = _decode_output(raw, max_bytes=10)
        assert "truncated" in result.lower()

    def test_no_truncation_when_under_limit(self) -> None:
        raw = b"hello"
        result = _decode_output(raw, max_bytes=100)
        assert "truncated" not in result
        assert result == "hello"

    def test_empty_bytes_returns_empty_string(self) -> None:
        assert _decode_output(b"", 1024) == ""

    def test_truncation_at_exact_limit(self) -> None:
        raw = b"a" * 10
        result = _decode_output(raw, max_bytes=10)
        # Exactly at limit — not truncated
        assert "truncated" not in result

    def test_unicode_characters_preserved(self) -> None:
        raw = "こんにちは".encode("utf-8")
        result = _decode_output(raw, max_bytes=len(raw) + 10)
        assert "こんにちは" in result


# ---------------------------------------------------------------------------
# execute_command — empty command
# ---------------------------------------------------------------------------


class TestExecuteCommandEmpty:
    def test_empty_command_returns_zero_exit(self) -> None:
        result = execute_command("")
        assert result.exit_code == 0

    def test_whitespace_only_returns_zero_exit(self) -> None:
        result = execute_command("   ")
        assert result.exit_code == 0

    def test_empty_command_no_stdout(self) -> None:
        result = execute_command("")
        assert result.stdout == ""

    def test_empty_command_zero_duration(self) -> None:
        result = execute_command("")
        assert result.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# execute_command — captured mode (stream_output=False)
# ---------------------------------------------------------------------------


class TestExecuteCommandCaptured:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_captures_stdout(self) -> None:
        result = execute_command(
            'echo "hello from test"',
            stream_output=False,
        )
        assert "hello from test" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_exit_code_zero_on_success(self) -> None:
        result = execute_command("true", stream_output=False)
        assert result.exit_code == 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_exit_code_nonzero_on_failure(self) -> None:
        result = execute_command("false", stream_output=False)
        assert result.exit_code != 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_captures_stderr(self) -> None:
        result = execute_command(
            "echo 'err msg' >&2",
            stream_output=False,
        )
        assert "err msg" in result.stderr

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_command_stored_in_result(self) -> None:
        cmd = "echo test"
        result = execute_command(cmd, stream_output=False)
        assert result.command == cmd

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_duration_positive(self) -> None:
        result = execute_command("echo hi", stream_output=False)
        assert result.duration_seconds >= 0.0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_multiline_output_captured(self) -> None:
        result = execute_command(
            "printf 'line1\nline2\nline3\n'",
            stream_output=False,
        )
        assert "line1" in result.stdout
        assert "line2" in result.stdout
        assert "line3" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_succeeded_property(self) -> None:
        result = execute_command("true", stream_output=False)
        assert result.succeeded is True

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_failed_property(self) -> None:
        result = execute_command("false", stream_output=False)
        assert result.failed is True


# ---------------------------------------------------------------------------
# execute_command — timeout handling
# ---------------------------------------------------------------------------


class TestExecuteCommandTimeout:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_raises_command_timeout_error(self) -> None:
        with pytest.raises(CommandTimeoutError):
            execute_command(
                "sleep 60",
                stream_output=False,
                timeout=1,
            )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_error_has_correct_timeout_value(self) -> None:
        with pytest.raises(CommandTimeoutError) as exc_info:
            execute_command("sleep 60", stream_output=False, timeout=1)
        assert exc_info.value.timeout == 1

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_no_timeout_completes_normally(self) -> None:
        result = execute_command(
            "echo done",
            stream_output=False,
            timeout=10,
        )
        assert result.succeeded

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_error_message_contains_command(self) -> None:
        cmd = "sleep 60"
        with pytest.raises(CommandTimeoutError) as exc_info:
            execute_command(cmd, stream_output=False, timeout=1)
        assert cmd in str(exc_info.value)


# ---------------------------------------------------------------------------
# execute_command — working directory
# ---------------------------------------------------------------------------


class TestExecuteCommandCwd:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_cwd_applied_to_subprocess(self, tmp_path) -> None:
        result = execute_command(
            "pwd",
            stream_output=False,
            cwd=str(tmp_path),
        )
        assert str(tmp_path) in result.stdout


# ---------------------------------------------------------------------------
# execute_command — error cases
# ---------------------------------------------------------------------------


class TestExecuteCommandErrors:
    def test_command_not_found_in_shell_returns_nonzero(self) -> None:
        # When running via shell, an unknown command produces a non-zero exit
        result = execute_command(
            "this_command_definitely_does_not_exist_12345",
            stream_output=False,
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CommandTimeoutError
# ---------------------------------------------------------------------------


class TestCommandTimeoutError:
    def test_attributes(self) -> None:
        err = CommandTimeoutError("sleep 10", 5)
        assert err.command == "sleep 10"
        assert err.timeout == 5

    def test_string_representation(self) -> None:
        err = CommandTimeoutError("sleep 10", 5)
        assert "5" in str(err)
        assert "sleep 10" in str(err)

    def test_is_executor_error(self) -> None:
        err = CommandTimeoutError("sleep", 1)
        assert isinstance(err, ExecutorError)


# ---------------------------------------------------------------------------
# CommandNotFoundError
# ---------------------------------------------------------------------------


class TestCommandNotFoundError:
    def test_attributes(self) -> None:
        err = CommandNotFoundError("nonexistent_cmd")
        assert err.command == "nonexistent_cmd"

    def test_string_representation(self) -> None:
        err = CommandNotFoundError("nonexistent_cmd")
        assert "nonexistent_cmd" in str(err)

    def test_is_executor_error(self) -> None:
        err = CommandNotFoundError("nonexistent_cmd")
        assert isinstance(err, ExecutorError)


# ---------------------------------------------------------------------------
# execute_command_with_output_callback
# ---------------------------------------------------------------------------


class TestExecuteCommandWithCallback:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_stdout_callback_invoked(self) -> None:
        lines: List[str] = []
        execute_command_with_output_callback(
            "printf 'a\nb\nc\n'",
            stdout_callback=lines.append,
        )
        assert "a" in lines
        assert "b" in lines
        assert "c" in lines

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_stderr_callback_invoked(self) -> None:
        lines: List[str] = []
        execute_command_with_output_callback(
            "echo 'error' >&2",
            stderr_callback=lines.append,
        )
        assert "error" in lines

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_stdout_in_result(self) -> None:
        result = execute_command_with_output_callback(
            "echo hello",
        )
        assert "hello" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_exit_code_captured(self) -> None:
        result = execute_command_with_output_callback("true")
        assert result.exit_code == 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_failure_exit_code(self) -> None:
        result = execute_command_with_output_callback("false")
        assert result.exit_code != 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_callback_exception_does_not_propagate(self) -> None:
        def bad_callback(line: str) -> None:
            raise RuntimeError("callback error")

        # Should not raise even if callback throws
        result = execute_command_with_output_callback(
            "echo hi",
            stdout_callback=bad_callback,
        )
        assert result is not None

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_command_stored_in_result(self) -> None:
        cmd = "echo test"
        result = execute_command_with_output_callback(cmd)
        assert result.command == cmd

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_raises(self) -> None:
        with pytest.raises(CommandTimeoutError):
            execute_command_with_output_callback(
                "sleep 60",
                timeout=1,
            )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_no_callbacks_still_captures(self) -> None:
        result = execute_command_with_output_callback("echo output")
        assert "output" in result.stdout
