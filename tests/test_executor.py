"""Unit tests for safe_run.executor module.

Covers command execution, timeout handling, output capture, streaming,
exit code handling, callback-based output, and error conditions.
"""

from __future__ import annotations

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

    def test_stdout_stored(self) -> None:
        result = ExecutionResult(command="echo hi", exit_code=0, stdout="hi\n")
        assert result.stdout == "hi\n"

    def test_stderr_stored(self) -> None:
        result = ExecutionResult(command="cmd", exit_code=1, stderr="error msg\n")
        assert result.stderr == "error msg\n"

    def test_interrupted_attribute(self) -> None:
        result = ExecutionResult(command="sleep 5", exit_code=-1, interrupted=True)
        assert result.interrupted is True

    def test_succeeded_and_failed_are_mutually_exclusive_zero(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.succeeded is True
        assert result.failed is False

    def test_succeeded_and_failed_are_mutually_exclusive_nonzero(self) -> None:
        result = ExecutionResult(command="false", exit_code=1)
        assert result.succeeded is False
        assert result.failed is True

    def test_default_duration_zero(self) -> None:
        result = ExecutionResult(command="ls", exit_code=0)
        assert result.duration_seconds == 0.0

    def test_large_exit_code(self) -> None:
        result = ExecutionResult(command="cmd", exit_code=255)
        assert result.exit_code == 255
        assert result.failed is True

    def test_negative_exit_code(self) -> None:
        result = ExecutionResult(command="cmd", exit_code=-1)
        assert result.exit_code == -1
        assert result.failed is True


# ---------------------------------------------------------------------------
# _decode_output
# ---------------------------------------------------------------------------


class TestDecodeOutput:
    def test_decodes_utf8(self) -> None:
        raw = b"hello world\n"
        assert _decode_output(raw, 1024) == "hello world\n"

    def test_decodes_latin1_fallback(self) -> None:
        # Bytes that are invalid UTF-8
        raw = bytes([0xFF, 0xFE, 0x41])
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

    def test_truncation_one_byte_over_limit(self) -> None:
        raw = b"a" * 11
        result = _decode_output(raw, max_bytes=10)
        assert "truncated" in result.lower()

    def test_newline_preserved_in_output(self) -> None:
        raw = b"line1\nline2\n"
        result = _decode_output(raw, max_bytes=1024)
        assert "line1" in result
        assert "line2" in result

    def test_binary_data_does_not_raise(self) -> None:
        # Random binary bytes that are not valid UTF-8
        raw = bytes(range(256))
        result = _decode_output(raw, max_bytes=len(raw) + 10)
        assert isinstance(result, str)

    def test_large_max_bytes_no_truncation(self) -> None:
        raw = b"hello world"
        result = _decode_output(raw, max_bytes=10 * 1024 * 1024)
        assert result == "hello world"
        assert "truncated" not in result

    def test_zero_max_bytes_always_truncated(self) -> None:
        raw = b"hello"
        result = _decode_output(raw, max_bytes=0)
        # Nothing kept, but truncation notice should be appended
        assert "truncated" in result.lower()


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

    def test_empty_command_no_stderr(self) -> None:
        result = execute_command("")
        assert result.stderr == ""

    def test_empty_command_timed_out_false(self) -> None:
        result = execute_command("")
        assert result.timed_out is False

    def test_empty_command_not_interrupted(self) -> None:
        result = execute_command("")
        assert result.interrupted is False

    def test_empty_command_preserves_command_field(self) -> None:
        result = execute_command("")
        assert result.command == ""

    def test_tab_only_command_returns_zero(self) -> None:
        result = execute_command("\t\t")
        assert result.exit_code == 0


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
            "printf 'line1\\nline2\\nline3\\n'",
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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_exit_code_1_from_false(self) -> None:
        result = execute_command("false", stream_output=False)
        assert result.exit_code == 1

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_exit_code_specific_value(self) -> None:
        result = execute_command("exit 42", stream_output=False, shell=True)
        assert result.exit_code == 42

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_no_output_when_command_produces_none(self) -> None:
        result = execute_command("true", stream_output=False)
        assert result.stdout == ""
        assert result.stderr == ""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_stdout_and_stderr_both_captured(self) -> None:
        result = execute_command(
            "echo stdout_msg; echo stderr_msg >&2",
            stream_output=False,
        )
        assert "stdout_msg" in result.stdout
        assert "stderr_msg" in result.stderr

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timed_out_false_on_fast_command(self) -> None:
        result = execute_command("echo quick", stream_output=False)
        assert result.timed_out is False

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_interrupted_false_on_normal_exit(self) -> None:
        result = execute_command("true", stream_output=False)
        assert result.interrupted is False


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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_error_message_contains_timeout_duration(self) -> None:
        with pytest.raises(CommandTimeoutError) as exc_info:
            execute_command("sleep 60", stream_output=False, timeout=2)
        assert "2" in str(exc_info.value)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_error_is_executor_error(self) -> None:
        with pytest.raises(ExecutorError):
            execute_command("sleep 60", stream_output=False, timeout=1)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_fast_command_completes_within_generous_timeout(self) -> None:
        result = execute_command(
            "echo fast",
            stream_output=False,
            timeout=30,
        )
        assert result.succeeded
        assert "fast" in result.stdout


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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_cwd_affects_file_creation(self, tmp_path) -> None:
        test_file = tmp_path / "test_output.txt"
        execute_command(
            "touch test_output.txt",
            stream_output=False,
            cwd=str(tmp_path),
        )
        assert test_file.exists()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_default_cwd_is_current_directory(self) -> None:
        import os
        result = execute_command("pwd", stream_output=False)
        # Should succeed and return some path
        assert result.succeeded
        assert result.stdout.strip() != ""


# ---------------------------------------------------------------------------
# execute_command — environment variables
# ---------------------------------------------------------------------------


class TestExecuteCommandEnv:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_custom_env_variable_visible_in_command(self) -> None:
        import os
        env = os.environ.copy()
        env["SAFE_RUN_TEST_VAR"] = "test_value_12345"
        result = execute_command(
            "echo $SAFE_RUN_TEST_VAR",
            stream_output=False,
            env=env,
        )
        assert "test_value_12345" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_none_env_inherits_parent_environment(self) -> None:
        import os
        os.environ["SAFE_RUN_INHERIT_TEST"] = "inherited_value"
        try:
            result = execute_command(
                "echo $SAFE_RUN_INHERIT_TEST",
                stream_output=False,
                env=None,
            )
            assert "inherited_value" in result.stdout
        finally:
            del os.environ["SAFE_RUN_INHERIT_TEST"]


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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_syntax_error_returns_nonzero(self) -> None:
        result = execute_command(
            "if; then; fi",
            stream_output=False,
        )
        assert result.exit_code != 0

    def test_result_is_execution_result_instance(self) -> None:
        result = execute_command("", stream_output=False)
        assert isinstance(result, ExecutionResult)


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

    def test_is_exception(self) -> None:
        err = CommandTimeoutError("sleep", 1)
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(CommandTimeoutError):
            raise CommandTimeoutError("sleep 60", 10)

    def test_can_be_caught_as_executor_error(self) -> None:
        with pytest.raises(ExecutorError):
            raise CommandTimeoutError("sleep 60", 10)

    def test_zero_timeout_stored(self) -> None:
        # Edge case: zero timeout
        err = CommandTimeoutError("cmd", 0)
        assert err.timeout == 0

    def test_command_with_spaces_stored(self) -> None:
        cmd = "rm -rf /tmp/test dir"
        err = CommandTimeoutError(cmd, 30)
        assert err.command == cmd


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

    def test_is_exception(self) -> None:
        err = CommandNotFoundError("missing")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(CommandNotFoundError):
            raise CommandNotFoundError("nonexistent")

    def test_can_be_caught_as_executor_error(self) -> None:
        with pytest.raises(ExecutorError):
            raise CommandNotFoundError("nonexistent")

    def test_empty_command_stored(self) -> None:
        err = CommandNotFoundError("")
        assert err.command == ""


# ---------------------------------------------------------------------------
# ExecutorError
# ---------------------------------------------------------------------------


class TestExecutorError:
    def test_is_exception(self) -> None:
        err = ExecutorError("some error")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        err = ExecutorError("setup failed")
        assert "setup failed" in str(err)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ExecutorError, match="test error"):
            raise ExecutorError("test error")


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
            "printf 'a\\nb\\nc\\n'",
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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_duration_is_non_negative(self) -> None:
        result = execute_command_with_output_callback("echo hi")
        assert result.duration_seconds >= 0.0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_result_is_execution_result_instance(self) -> None:
        result = execute_command_with_output_callback("true")
        assert isinstance(result, ExecutionResult)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_multiline_stdout_callback_called_per_line(self) -> None:
        lines: List[str] = []
        execute_command_with_output_callback(
            "printf 'first\\nsecond\\nthird\\n'",
            stdout_callback=lines.append,
        )
        assert len(lines) >= 3
        assert "first" in lines
        assert "second" in lines
        assert "third" in lines

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_stderr_captured_in_result(self) -> None:
        result = execute_command_with_output_callback(
            "echo error_line >&2",
        )
        assert "error_line" in result.stderr

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_both_callbacks_invoked_independently(self) -> None:
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        execute_command_with_output_callback(
            "echo out_line; echo err_line >&2",
            stdout_callback=stdout_lines.append,
            stderr_callback=stderr_lines.append,
        )
        assert any("out_line" in l for l in stdout_lines)
        assert any("err_line" in l for l in stderr_lines)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_cwd_parameter_respected(self, tmp_path) -> None:
        result = execute_command_with_output_callback(
            "pwd",
            cwd=str(tmp_path),
        )
        assert str(tmp_path) in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_timeout_error_is_command_timeout_error(self) -> None:
        with pytest.raises(CommandTimeoutError) as exc_info:
            execute_command_with_output_callback(
                "sleep 60",
                timeout=1,
            )
        assert exc_info.value.timeout == 1

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_succeeded_property_on_result(self) -> None:
        result = execute_command_with_output_callback("true")
        assert result.succeeded is True

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_failed_property_on_result(self) -> None:
        result = execute_command_with_output_callback("false")
        assert result.failed is True


# ---------------------------------------------------------------------------
# _terminate_process
# ---------------------------------------------------------------------------


class TestTerminateProcess:
    def test_terminate_already_finished_process_does_not_raise(self) -> None:
        import subprocess
        # Start and immediately finish a process
        proc = subprocess.Popen(
            "true",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()  # Let it finish
        # Terminating an already-finished process should not raise
        _terminate_process(proc)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only signal handling"
    )
    def test_terminate_running_process_stops_it(self) -> None:
        import subprocess
        proc = subprocess.Popen(
            "sleep 60",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Process should be running
        assert proc.poll() is None
        _terminate_process(proc)
        # Give it a moment to terminate
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        assert proc.returncode is not None


# ---------------------------------------------------------------------------
# execute_command — streaming mode (stream_output=True)
# ---------------------------------------------------------------------------


class TestExecuteCommandStreaming:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_exit_code_zero_on_success(self) -> None:
        result = execute_command("true", stream_output=True)
        assert result.exit_code == 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_exit_code_nonzero_on_failure(self) -> None:
        result = execute_command("false", stream_output=True)
        assert result.exit_code != 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_stdout_is_empty_string(self) -> None:
        # Streaming mode doesn't capture stdout — it goes to terminal
        result = execute_command("echo hi", stream_output=True)
        assert result.stdout == ""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_stderr_is_empty_string(self) -> None:
        result = execute_command("echo err >&2", stream_output=True)
        assert result.stderr == ""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_command_stored_in_result(self) -> None:
        cmd = "echo streaming_test"
        result = execute_command(cmd, stream_output=True)
        assert result.command == cmd

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_timed_out_false_on_fast_command(self) -> None:
        result = execute_command("echo quick", stream_output=True)
        assert result.timed_out is False

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_timeout_raises(self) -> None:
        with pytest.raises(CommandTimeoutError):
            execute_command(
                "sleep 60",
                stream_output=True,
                timeout=1,
            )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_streaming_duration_is_non_negative(self) -> None:
        result = execute_command("true", stream_output=True)
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# execute_command — output truncation
# ---------------------------------------------------------------------------


class TestExecuteCommandOutputTruncation:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_truncates_large_stdout(self) -> None:
        # Generate ~500 bytes of output but limit to 100 bytes
        result = execute_command(
            "python3 -c \"print('x' * 500)\"",
            stream_output=False,
            max_output_bytes=100,
        )
        assert "truncated" in result.stdout.lower()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_does_not_truncate_small_stdout(self) -> None:
        result = execute_command(
            'echo "small output"',
            stream_output=False,
            max_output_bytes=10 * 1024 * 1024,
        )
        assert "small output" in result.stdout
        assert "truncated" not in result.stdout.lower()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestExecuteCommandIntegration:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_pipeline_command_captured(self) -> None:
        result = execute_command(
            "echo 'hello world' | tr 'a-z' 'A-Z'",
            stream_output=False,
        )
        assert "HELLO WORLD" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_command_with_environment_substitution(self) -> None:
        import os
        result = execute_command(
            "echo $HOME",
            stream_output=False,
        )
        assert os.environ.get("HOME", "") in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_sequential_commands_with_semicolon(self) -> None:
        result = execute_command(
            "echo first; echo second",
            stream_output=False,
        )
        assert "first" in result.stdout
        assert "second" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_conditional_and_operator(self) -> None:
        result = execute_command(
            "true && echo 'and_worked'",
            stream_output=False,
        )
        assert "and_worked" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_conditional_and_short_circuit_on_failure(self) -> None:
        result = execute_command(
            "false && echo 'should_not_appear'",
            stream_output=False,
        )
        assert "should_not_appear" not in result.stdout
        assert result.exit_code != 0

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_or_operator_on_failure(self) -> None:
        result = execute_command(
            "false || echo 'or_worked'",
            stream_output=False,
        )
        assert "or_worked" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_subshell_command(self) -> None:
        result = execute_command(
            "(echo subshell_output)",
            stream_output=False,
        )
        assert "subshell_output" in result.stdout

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX-only commands"
    )
    def test_command_with_file_redirection(self, tmp_path) -> None:
        outfile = tmp_path / "output.txt"
        result = execute_command(
            f"echo file_content > {outfile}",
            stream_output=False,
            cwd=str(tmp_path),
        )
        assert result.succeeded
        assert outfile.exists()
        assert "file_content" in outfile.read_text()
