"""Safe subprocess command executor for safe_run.

This module provides functionality to execute confirmed shell commands via
subprocess with streaming output, timeout handling, and proper error reporting.
It supports both real-time output streaming and captured execution modes.

Typical usage example::

    from safe_run.executor import execute_command, ExecutionResult

    result = execute_command("ls -la /tmp")
    print(result.exit_code)
    print(result.stdout)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import IO, List, Optional


# Default timeout for command execution in seconds (None = no timeout)
DEFAULT_TIMEOUT: Optional[int] = None

# Default maximum output buffer size in bytes (10 MB)
DEFAULT_MAX_OUTPUT_BYTES: int = 10 * 1024 * 1024


class ExecutorError(Exception):
    """Raised when the executor encounters an unrecoverable setup error."""


class CommandTimeoutError(ExecutorError):
    """Raised when a command exceeds its allowed execution time."""

    def __init__(self, command: str, timeout: int) -> None:
        self.command = command
        self.timeout = timeout
        super().__init__(
            f"Command timed out after {timeout} seconds: {command!r}"
        )


class CommandNotFoundError(ExecutorError):
    """Raised when the shell or command executable cannot be found."""

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(f"Command not found or shell unavailable: {command!r}")


@dataclass
class ExecutionResult:
    """The result of executing a shell command.

    Attributes:
        command: The original command string that was executed.
        exit_code: The process exit code. 0 typically means success.
        stdout: Captured standard output as a string. Empty if streaming
            mode was used without capture.
        stderr: Captured standard error as a string. Empty if streaming
            mode was used without capture.
        timed_out: True if the command was killed due to a timeout.
        duration_seconds: Wall-clock time the command took to complete.
        interrupted: True if the command was interrupted by the user
            (e.g. Ctrl+C / SIGINT).
    """

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    interrupted: bool = False

    @property
    def succeeded(self) -> bool:
        """Return True if the command exited with code 0.

        Returns:
            True when exit_code == 0.
        """
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        """Return True if the command exited with a non-zero code.

        Returns:
            True when exit_code != 0.
        """
        return not self.succeeded


def execute_command(
    command: str,
    *,
    stream_output: bool = True,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    shell: bool = True,
) -> ExecutionResult:
    """Execute a shell command and return the result.

    Runs the command in a subprocess using the system shell (``/bin/sh -c``
    on POSIX, ``cmd /c`` on Windows). Supports real-time output streaming
    and optional timeout enforcement.

    When ``stream_output=True``, stdout and stderr are forwarded directly
    to the calling process's stdout/stderr in real time. The returned
    ``ExecutionResult.stdout`` and ``ExecutionResult.stderr`` will be empty
    strings in this mode.

    When ``stream_output=False``, output is captured and returned in the
    ``ExecutionResult``.

    Args:
        command: The shell command string to execute.
        stream_output: If True (default), stream stdout and stderr to the
            terminal in real time. If False, capture them.
        timeout: Maximum number of seconds to allow the command to run.
            None means no timeout (default).
        cwd: Working directory for the subprocess. Defaults to the current
            directory of the parent process.
        env: Environment variables for the subprocess. If None, inherits
            the parent process environment.
        max_output_bytes: Maximum bytes to capture from stdout+stderr when
            stream_output=False. Excess output is silently truncated.
        shell: Whether to run via the system shell. Default True.

    Returns:
        An ExecutionResult with the exit code, output, and metadata.

    Raises:
        ExecutorError: If subprocess cannot be launched (e.g. shell not found).
        CommandTimeoutError: If the command exceeds the timeout.
    """
    import time

    if not command or not command.strip():
        return ExecutionResult(
            command=command,
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    start_time = time.monotonic()

    if stream_output:
        result = _execute_streaming(
            command=command,
            timeout=timeout,
            cwd=cwd,
            env=env,
            shell=shell,
        )
    else:
        result = _execute_captured(
            command=command,
            timeout=timeout,
            cwd=cwd,
            env=env,
            shell=shell,
            max_output_bytes=max_output_bytes,
        )

    elapsed = time.monotonic() - start_time
    # Return a new dataclass with the duration filled in
    return ExecutionResult(
        command=result.command,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        duration_seconds=elapsed,
        interrupted=result.interrupted,
    )


def _execute_streaming(
    command: str,
    timeout: Optional[int],
    cwd: Optional[str],
    env: Optional[dict],
    shell: bool,
) -> ExecutionResult:
    """Execute a command while streaming output directly to the terminal.

    Stdout and stderr are inherited from the parent process so the child's
    output appears in the terminal in real time, including ANSI colours and
    interactive prompts.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds, or None.
        cwd: Working directory for the child process.
        env: Environment variables, or None to inherit.
        shell: Whether to run via the system shell.

    Returns:
        An ExecutionResult with stdout/stderr as empty strings (not captured).

    Raises:
        ExecutorError: If the process cannot be launched.
        CommandTimeoutError: If the command exceeds the timeout.
    """
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            # Inherit stdin/stdout/stderr from parent for full interactivity
            stdin=None,
            stdout=None,
            stderr=None,
        )
    except FileNotFoundError as exc:
        raise CommandNotFoundError(command) from exc
    except OSError as exc:
        raise ExecutorError(
            f"Failed to launch command {command!r}: {exc}"
        ) from exc

    timed_out = False
    interrupted = False

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
    except KeyboardInterrupt:
        interrupted = True
        _terminate_process(process)

    exit_code = process.returncode if process.returncode is not None else -1

    if timed_out:
        raise CommandTimeoutError(command, timeout)  # type: ignore[arg-type]

    return ExecutionResult(
        command=command,
        exit_code=exit_code,
        stdout="",
        stderr="",
        timed_out=timed_out,
        interrupted=interrupted,
    )


def _execute_captured(
    command: str,
    timeout: Optional[int],
    cwd: Optional[str],
    env: Optional[dict],
    shell: bool,
    max_output_bytes: int,
) -> ExecutionResult:
    """Execute a command and capture its stdout and stderr.

    Output is collected into memory. If the combined output exceeds
    ``max_output_bytes``, the excess is truncated with a notice appended.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds, or None.
        cwd: Working directory for the child process.
        env: Environment variables, or None to inherit.
        shell: Whether to run via the system shell.
        max_output_bytes: Maximum bytes to retain from stdout and stderr.

    Returns:
        An ExecutionResult with captured stdout and stderr.

    Raises:
        ExecutorError: If the process cannot be launched.
        CommandTimeoutError: If the command exceeds the timeout.
    """
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CommandNotFoundError(command) from exc
    except OSError as exc:
        raise ExecutorError(
            f"Failed to launch command {command!r}: {exc}"
        ) from exc

    timed_out = False
    interrupted = False

    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
        except Exception:
            stdout_bytes, stderr_bytes = b"", b""
        raise CommandTimeoutError(command, timeout)  # type: ignore[arg-type]
    except KeyboardInterrupt:
        interrupted = True
        _terminate_process(process)
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
        except Exception:
            stdout_bytes, stderr_bytes = b"", b""

    exit_code = process.returncode if process.returncode is not None else -1

    stdout_str = _decode_output(stdout_bytes, max_output_bytes)
    stderr_str = _decode_output(stderr_bytes, max_output_bytes)

    return ExecutionResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout_str,
        stderr=stderr_str,
        timed_out=timed_out,
        interrupted=interrupted,
    )


def _decode_output(raw: bytes, max_bytes: int) -> str:
    """Decode raw bytes output from a subprocess to a string.

    Attempts UTF-8 decoding first; falls back to latin-1 (which never fails).
    Truncates output that exceeds max_bytes before decoding.

    Args:
        raw: Raw bytes from subprocess stdout or stderr.
        max_bytes: Maximum number of bytes to include before truncating.

    Returns:
        A decoded string, possibly with a truncation notice appended.
    """
    truncated = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if truncated:
        text += "\n[output truncated — exceeded maximum buffer size]"

    return text


def _terminate_process(process: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Attempt to gracefully terminate a subprocess, then force-kill it.

    Sends SIGTERM (or CTRL_BREAK_EVENT on Windows) first. If the process
    does not exit within 3 seconds, sends SIGKILL (or TerminateProcess).

    Args:
        process: The subprocess.Popen instance to terminate.
    """
    try:
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except OSError:
        # Process may have already exited
        return

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2)
        except OSError:
            pass


def execute_command_with_output_callback(
    command: str,
    *,
    stdout_callback=None,
    stderr_callback=None,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    shell: bool = True,
) -> ExecutionResult:
    """Execute a command and invoke callbacks for each line of output.

    This variant allows the caller to process output line-by-line as it is
    produced, for example to feed it into a Rich Live display. Stdout and
    stderr are read in separate threads to avoid deadlocks.

    Args:
        command: The shell command string to execute.
        stdout_callback: Optional callable that receives each stdout line
            (as a string, stripped of the trailing newline).
        stderr_callback: Optional callable that receives each stderr line
            (as a string, stripped of the trailing newline).
        timeout: Maximum execution time in seconds, or None.
        cwd: Working directory for the child process.
        env: Environment variables, or None to inherit.
        shell: Whether to run via the system shell.

    Returns:
        An ExecutionResult. The stdout and stderr fields contain the full
        collected output as strings.

    Raises:
        ExecutorError: If the process cannot be launched.
        CommandTimeoutError: If the command exceeds the timeout.
    """
    import time

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError as exc:
        raise CommandNotFoundError(command) from exc
    except OSError as exc:
        raise ExecutorError(
            f"Failed to launch command {command!r}: {exc}"
        ) from exc

    def _read_stream(
        stream: IO[str],
        lines: List[str],
        callback,
    ) -> None:
        """Read lines from a stream and optionally invoke a callback."""
        try:
            for line in stream:
                stripped = line.rstrip("\n")
                lines.append(stripped)
                if callback is not None:
                    try:
                        callback(stripped)
                    except Exception:
                        pass
        except (OSError, ValueError):
            pass

    stdout_thread = threading.Thread(
        target=_read_stream,
        args=(process.stdout, stdout_lines, stdout_callback),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream,
        args=(process.stderr, stderr_lines, stderr_callback),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    interrupted = False
    start_time = time.monotonic()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
    except KeyboardInterrupt:
        interrupted = True
        _terminate_process(process)

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    elapsed = time.monotonic() - start_time
    exit_code = process.returncode if process.returncode is not None else -1

    if timed_out:
        raise CommandTimeoutError(command, timeout)  # type: ignore[arg-type]

    return ExecutionResult(
        command=command,
        exit_code=exit_code,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
        timed_out=timed_out,
        duration_seconds=elapsed,
        interrupted=interrupted,
    )
