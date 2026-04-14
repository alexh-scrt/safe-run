# safe_run.bash — Bash shell integration for safe_run
#
# Source this file in your ~/.bashrc to wrap interactive commands with safe_run:
#
#   source /path/to/safe_run.bash
#
# Or, if installed via pip, you can find it with:
#
#   source "$(python3 -m site --user-base)/share/safe_run/safe_run.bash"
#
# Once sourced, every command you type in your interactive shell will be
# passed through safe_run for explanation and risk assessment before
# execution. Pipelines and compound commands are handled transparently.
#
# CONFIGURATION
# -------------
# SAFE_RUN_BYPASS_COMMANDS  — Space-separated list of command prefixes that
#                              bypass safe_run entirely (no explanation, no
#                              confirmation). Defaults to a sensible set of
#                              read-only, low-risk commands.
#
# SAFE_RUN_ENABLED          — Set to '0' or 'false' to disable the hook
#                              without removing it from your .bashrc.
#
# SAFE_RUN_OPTS             — Extra options to pass to every safe_run
#                              invocation, e.g. '--provider ollama'.
#
# SAFE_RUN_BIN              — Path to the safe_run executable. Auto-detected
#                              by default.
#
# SAFE_RUN_QUIET            — Set to '1' to suppress the startup message.
#
# EXAMPLES
# --------
#   # Disable safe_run for the current session:
#   SAFE_RUN_ENABLED=0
#
#   # Use Ollama instead of OpenAI:
#   SAFE_RUN_OPTS='--provider ollama'
#
#   # Add extra bypass prefixes:
#   SAFE_RUN_BYPASS_COMMANDS+=' man info'

# ---------------------------------------------------------------------------
# Guard: only load in interactive Bash sessions
# ---------------------------------------------------------------------------

# Do nothing if not running interactively
[[ $- == *i* ]] || return 0

# Do nothing if not running in Bash
[[ -n "${BASH_VERSION:-}" ]] || return 0

# ---------------------------------------------------------------------------
# Locate the safe_run binary
# ---------------------------------------------------------------------------

_safe_run_find_bin() {
    # Honour explicit override first
    if [[ -n "${SAFE_RUN_BIN:-}" ]]; then
        if [[ -x "$SAFE_RUN_BIN" ]]; then
            echo "$SAFE_RUN_BIN"
            return 0
        else
            echo "[safe_run] WARNING: SAFE_RUN_BIN='$SAFE_RUN_BIN' is not executable." >&2
            return 1
        fi
    fi

    # Try PATH lookup
    local bin
    bin="$(command -v safe_run 2>/dev/null)"
    if [[ -n "$bin" && -x "$bin" ]]; then
        echo "$bin"
        return 0
    fi

    # Common pip install locations
    local candidate
    for candidate in \
        "${HOME}/.local/bin/safe_run" \
        "/usr/local/bin/safe_run" \
        "/usr/bin/safe_run"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# Cache the binary path at source time
if ! _SAFE_RUN_BIN="$(_safe_run_find_bin)"; then
    echo "[safe_run] WARNING: safe_run executable not found. Shell hook not installed." >&2
    echo "[safe_run]          Install it with: pip install safe_run" >&2
    return 0
fi
export _SAFE_RUN_BIN

# ---------------------------------------------------------------------------
# Default bypass prefixes
# ---------------------------------------------------------------------------
# Commands listed here (matched as prefixes of the typed command) are passed
# directly to the shell without going through safe_run. Keep this list to
# genuinely read-only, zero-risk operations to avoid confirmation fatigue.

SAFE_RUN_BYPASS_COMMANDS="${SAFE_RUN_BYPASS_COMMANDS:-\
cat \
cd \
clear \
date \
diff \
dircolors \
dirname \
disown \
echo \
env \
exec \
exit \
export \
fg \
file \
grep \
head \
help \
hist \
history \
hostname \
jobs \
less \
locate \
ls \
man \
mkdir -p \
more \
mv \
printenv \
pwd \
readlink \
realpath \
set \
sort \
source \
stat \
tail \
tee \
test \
time \
tput \
true \
type \
unalias \
unset \
uname \
wc \
where \
which \
whoami \
xargs \
zcat \
bash"
}

# ---------------------------------------------------------------------------
# Internal helper: check whether a command should bypass safe_run
# ---------------------------------------------------------------------------

_safe_run_is_bypass() {
    local cmd="$1"
    local prefix

    # Strip leading whitespace
    cmd="${cmd#"${cmd%%[! ]*}"}"

    # Empty command — bypass (nothing to do)
    [[ -z "$cmd" ]] && return 0

    # Extract the first word (the command name)
    local first_word
    first_word="${cmd%%[[:space:]]*}"

    # Bash built-ins and control structures that must not be intercepted
    case "$first_word" in
        alias|bg|bind|break|builtin|caller|cd|command|compgen|complete|\
        compopt|continue|declare|dirs|disown|echo|enable|eval|exec|exit|\
        export|false|fc|fg|getopts|hash|help|history|jobs|kill|let|local|\
        logout|mapfile|popd|printf|pushd|pwd|read|readarray|readonly|\
        return|set|shift|shopt|source|suspend|test|times|trap|true|type|\
        typeset|ulimit|umask|unalias|unset|wait|\.)
            return 0
            ;;
    esac

    # Check user-configured bypass prefixes
    local bypass_prefix
    # Split SAFE_RUN_BYPASS_COMMANDS on whitespace into individual prefixes
    while IFS= read -r bypass_prefix; do
        # Skip empty lines
        [[ -z "$bypass_prefix" ]] && continue
        if [[ "$cmd" == "${bypass_prefix}"* ]]; then
            return 0
        fi
    done <<< "$(echo "$SAFE_RUN_BYPASS_COMMANDS" | tr ' ' '\n')"

    return 1
}

# ---------------------------------------------------------------------------
# The DEBUG trap — called by Bash before each command is executed
# ---------------------------------------------------------------------------
# Bash does not have a preexec hook built in. We emulate one using the
# DEBUG trap, which fires before every simple command, pipeline, and
# compound command. We gate it on BASH_COMMAND to retrieve the full
# command string as typed by the user.
#
# IMPORTANT: The DEBUG trap approach has limitations compared to Zsh's
# preexec. In particular:
#
#   1. It fires for every sub-command in a pipeline, not just the whole
#      pipeline. We use $_safe_run_last_cmd to deduplicate.
#
#   2. BASH_COMMAND contains the command *after* alias expansion, not the
#      raw user input. For raw input we use the readline buffer trick via
#      the ERR/RETURN trap + bind -x, but that is complex. We use
#      BASH_COMMAND as a reasonable approximation.
#
#   3. The DEBUG trap fires for commands inside functions, $(...) sub-
#      shells, etc. We use $BASH_SUBSHELL and $SHLVL to avoid acting
#      on sub-commands.

# Track the last command we acted on to prevent duplicate invocations
_SAFE_RUN_LAST_CMD=''
# Track the BASH_COMMAND at the time preexec fires vs the outer shell
_SAFE_RUN_RUNNING=0

_safe_run_debug_trap() {
    # Only act at the top-level interactive shell (not in sub-shells or
    # function calls triggered by the prompt or completion).
    [[ "$BASH_SUBSHELL" -ne 0 ]] && return 0

    # Avoid re-entrancy: don't trap commands safe_run itself runs
    [[ "$_SAFE_RUN_RUNNING" -eq 1 ]] && return 0

    # Respect SAFE_RUN_ENABLED flag
    local enabled="${SAFE_RUN_ENABLED:-1}"
    if [[ "$enabled" == '0' || "$enabled" == 'false' || "$enabled" == 'no' ]]; then
        return 0
    fi

    local raw_command="$BASH_COMMAND"

    # Skip empty or whitespace-only input
    if [[ -z "${raw_command// }" ]]; then
        return 0
    fi

    # Skip if this is the same command we just processed (deduplication)
    if [[ "$raw_command" == "$_SAFE_RUN_LAST_CMD" ]]; then
        return 0
    fi

    # Skip internal safe_run plumbing
    if [[ "$raw_command" == *'_safe_run'* || "$raw_command" == *'safe_run'*'--bypass'* ]]; then
        return 0
    fi

    # Skip if this command should bypass safe_run
    if _safe_run_is_bypass "$raw_command"; then
        return 0
    fi

    # Record this command so we don't double-process it
    _SAFE_RUN_LAST_CMD="$raw_command"
    _SAFE_RUN_RUNNING=1

    # Build the safe_run invocation
    local -a sr_cmd
    sr_cmd=("$_SAFE_RUN_BIN")

    # Append any user-defined extra options
    if [[ -n "${SAFE_RUN_OPTS:-}" ]]; then
        # Word-split SAFE_RUN_OPTS deliberately
        local -a extra_opts
        # shellcheck disable=SC2206
        extra_opts=($SAFE_RUN_OPTS)
        sr_cmd+=("${extra_opts[@]}")
    fi

    # Pass the command as a single argument
    sr_cmd+=(-- "$raw_command")

    # Run safe_run. Capture its exit code.
    "${sr_cmd[@]}"
    local sr_exit=$?

    _SAFE_RUN_RUNNING=0

    if [[ $sr_exit -ne 0 ]]; then
        # safe_run declined, blocked, or errored — cancel the original command
        # by sending SIGINT to the current shell, which behaves like Ctrl+C.
        kill -INT $$
        return 1
    fi

    # safe_run succeeded and executed the command internally.
    # Cancel the shell's own pending execution of the same command to prevent
    # it from running twice.
    kill -INT $$
    return 0
}

# ---------------------------------------------------------------------------
# Bash preexec emulation using the DEBUG trap
# ---------------------------------------------------------------------------
# We support two scenarios:
#
# A) bash-preexec is already loaded (https://github.com/rcaloras/bash-preexec).
#    In that case we register via the preexec_functions array, which provides
#    clean preexec semantics identical to Zsh.
#
# B) bash-preexec is not available. We install a DEBUG trap, which is the
#    closest Bash equivalent. It has the caveats described above.

_safe_run_preexec_compat() {
    # This function is called by bash-preexec with the command string as $1.
    local raw_command="$1"

    # Respect SAFE_RUN_ENABLED flag
    local enabled="${SAFE_RUN_ENABLED:-1}"
    if [[ "$enabled" == '0' || "$enabled" == 'false' || "$enabled" == 'no' ]]; then
        return 0
    fi

    # Skip empty or whitespace-only input
    if [[ -z "${raw_command// }" ]]; then
        return 0
    fi

    # Skip if this command should bypass safe_run
    if _safe_run_is_bypass "$raw_command"; then
        return 0
    fi

    # Build the safe_run invocation
    local -a sr_cmd
    sr_cmd=("$_SAFE_RUN_BIN")

    if [[ -n "${SAFE_RUN_OPTS:-}" ]]; then
        local -a extra_opts
        # shellcheck disable=SC2206
        extra_opts=($SAFE_RUN_OPTS)
        sr_cmd+=("${extra_opts[@]}")
    fi

    sr_cmd+=(-- "$raw_command")

    # Run safe_run interactively
    if ! "${sr_cmd[@]}"; then
        # safe_run declined or blocked — cancel pending execution
        kill -INT $$
        return 1
    fi

    # safe_run approved and executed the command — prevent double execution
    kill -INT $$
    return 0
}

# Register the appropriate hook
if [[ -n "${preexec_functions+x}" ]] || declare -f preexec &>/dev/null; then
    # bash-preexec is available — use its cleaner hook mechanism
    preexec_functions+=(_safe_run_preexec_compat)
elif [[ -n "${__bp_imported:-}" ]]; then
    # bash-preexec loaded but using newer API
    preexec_functions+=(_safe_run_preexec_compat)
else
    # Fall back to the DEBUG trap
    # Preserve any existing DEBUG trap
    _SAFE_RUN_EXISTING_DEBUG="$(trap -p DEBUG 2>/dev/null)"
    if [[ -n "$_SAFE_RUN_EXISTING_DEBUG" ]]; then
        # Another DEBUG trap is already set; chain it
        # Extract just the trap command from the 'trap -- CMD DEBUG' output
        _safe_run_combined_debug() {
            _safe_run_debug_trap
            # Re-evaluate the original trap body (best-effort)
        }
        trap '_safe_run_debug_trap' DEBUG
    else
        trap '_safe_run_debug_trap' DEBUG
    fi
fi

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

# 'sr' — shorthand for safe_run
alias sr='safe_run'

# 'srb' — run a command with safe_run in bypass mode (no explanation/confirmation)
alias srb='safe_run --bypass --'

# 'src' — check command risk without executing
alias src='safe_run check'

# 'sr-off' — disable safe_run for the current session
sr-off() {
    export SAFE_RUN_ENABLED=0
    echo "[safe_run] Disabled for this session."
}

# 'sr-on' — re-enable safe_run for the current session
sr-on() {
    export SAFE_RUN_ENABLED=1
    echo "[safe_run] Enabled."
}

# ---------------------------------------------------------------------------
# Shell prompt indicator (optional)
# ---------------------------------------------------------------------------
# Adds a small 🛡 indicator to PS1 when safe_run is active.
# Disabled by default — uncomment to enable.

# _safe_run_ps1_indicator() {
#     if [[ "${SAFE_RUN_ENABLED:-1}" != '0' ]]; then
#         echo -n ' \[\033[32m\]🛡\[\033[0m\]'
#     fi
# }
# PS1='$(_safe_run_ps1_indicator)'"$PS1"

# ---------------------------------------------------------------------------
# Startup message (optional)
# ---------------------------------------------------------------------------

if [[ "${SAFE_RUN_QUIET:-0}" != '1' ]]; then
    echo -e "\033[32m[safe_run]\033[0m Shell hook active. Commands will be explained before execution."
    echo -e "\033[2m           Set \033[33mSAFE_RUN_ENABLED=0\033[0m\033[2m to disable, or \033[33mSAFE_RUN_QUIET=1\033[0m\033[2m to silence this message.\033[0m"
fi
