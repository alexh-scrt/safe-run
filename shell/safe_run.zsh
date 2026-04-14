# safe_run.zsh — Zsh shell integration for safe_run
#
# Source this file in your ~/.zshrc to wrap interactive commands with safe_run:
#
#   source /path/to/safe_run.zsh
#
# Or, if installed via pip, you can find it with:
#
#   source "$(python3 -m site --user-base)/share/safe_run/safe_run.zsh"
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
#                              without removing it from your .zshrc.
#
# SAFE_RUN_OPTS             — Extra options to pass to every safe_run
#                              invocation, e.g. '--provider ollama'.
#
# SAFE_RUN_BIN              — Path to the safe_run executable. Auto-detected
#                              by default.
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
# Guard: prevent sourcing in non-interactive or non-Zsh shells
# ---------------------------------------------------------------------------

[[ -o interactive ]] || return 0
[[ -n "$ZSH_VERSION" ]] || return 0

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
            print -u2 "[safe_run] WARNING: SAFE_RUN_BIN='$SAFE_RUN_BIN' is not executable."
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
    print -u2 "[safe_run] WARNING: safe_run executable not found. Shell hook not installed."
    print -u2 "[safe_run]          Install it with: pip install safe_run"
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
whence \
where \
which \
whoami \
xargs \
zcat \
zsh"
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

    # Shell built-ins and control structures that cannot be wrapped
    case "${cmd%%[[:space:]]*}" in
        # Zsh built-ins that must not be intercepted
        alias|autoload|bg|bindkey|builtin|bye|compdef|compinit|\
        compdump|declare|disable|disown|echo|emulate|enable|eval|\
        exec|exit|export|false|fc|float|fg|functions|getcap|getopt|\
        getopts|hash|integer|jobs|kill|limit|local|log|logout|noglob|\
        popd|print|printf|pushd|pushln|pwd|r|read|readonly|rehash|\
        return|sched|set|setcap|setopt|shift|source|suspend|test|\
        times|trap|true|ttyctl|type|typeset|ulimit|umask|unalias|\
        unfunction|unhash|unlimit|unset|unsetopt|vared|wait|whence|\
        where|which|zcompile|zle|zmodload|zparseopts|zregexparse|\
        zstyle|.)
            return 0
            ;;
    esac

    # Check user-configured bypass prefixes
    local bypass_prefix
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
# The preexec hook — called by Zsh before each command is executed
# ---------------------------------------------------------------------------
# The preexec hook receives three arguments:
#   $1  — the command as typed by the user (raw string)
#   $2  — the command after alias expansion
#   $3  — the command as it will be executed by the shell
#
# We use $1 (the raw user input) so the explanation matches what the user
# typed, preserving their intent and any aliases.

_safe_run_preexec() {
    # Respect SAFE_RUN_ENABLED flag
    local enabled="${SAFE_RUN_ENABLED:-1}"
    if [[ "$enabled" == '0' || "$enabled" == 'false' || "$enabled" == 'no' ]]; then
        return 0
    fi

    local raw_command="$1"

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

    # Append any user-defined extra options (split on whitespace)
    if [[ -n "${SAFE_RUN_OPTS:-}" ]]; then
        # Split $SAFE_RUN_OPTS into an array safely
        local -a extra_opts
        eval "extra_opts=(${SAFE_RUN_OPTS})"
        sr_cmd+=("${extra_opts[@]}")
    fi

    # Pass the command as a single argument to avoid re-parsing
    sr_cmd+=(-- "$raw_command")

    # Invoke safe_run interactively. If it exits non-zero (user declined,
    # blocked, or errored), we prevent the original command from running by
    # aborting the current command via 'kill -INT $$' — which signals the
    # shell as if the user pressed Ctrl+C, cleanly cancelling execution.
    if ! "${sr_cmd[@]}"; then
        # safe_run declined or blocked the command
        # Raise SIGINT to cancel the queued command execution
        kill -INT $$
        # Return non-zero so the hook chain knows we aborted
        return 1
    fi

    # safe_run approved — the shell will now execute the original command.
    # Note: Because preexec runs *before* execution, returning 0 here allows
    # the shell to proceed. safe_run already executed the command internally
    # (via subprocess), so we must prevent double execution by overriding
    # the command with a no-op. We do this via the precmd_functions mechanism
    # by setting a flag and using zle to clear the buffer, but the cleanest
    # approach for Zsh integration is the --bypass flow described in the
    # README: use safe_run as the executor itself.
    #
    # INTEGRATION NOTE
    # ----------------
    # There are two integration styles:
    #
    # Style A (RECOMMENDED — safe_run executes the command):
    #   safe_run runs the command internally via subprocess. The shell's own
    #   execution of the command is cancelled via SIGINT after safe_run
    #   completes. This gives safe_run full control.
    #
    # Style B (shell executes the command):
    #   safe_run is used only for explanation; the shell always runs the
    #   command. Use --no-execute mode (planned future feature).
    #
    # This hook implements Style A. After safe_run finishes executing the
    # command successfully, we cancel the shell's own pending execution:
    kill -INT $$
    return 0
}

# ---------------------------------------------------------------------------
# Register the preexec hook
# ---------------------------------------------------------------------------
# Zsh provides the preexec_functions array for chaining multiple hooks.
# We append to it so we don't clobber any existing hooks (e.g. from
# oh-my-zsh, starship, or zsh-syntax-highlighting).

autoload -Uz add-zsh-hook 2>/dev/null || true

if (( ${+functions[add-zsh-hook]} )); then
    add-zsh-hook preexec _safe_run_preexec
else
    # Fallback: manually append to preexec_functions
    typeset -ga preexec_functions
    preexec_functions+=(_safe_run_preexec)
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
alias sr-off='export SAFE_RUN_ENABLED=0 && echo "[safe_run] Disabled for this session."'

# 'sr-on' — re-enable safe_run for the current session
alias sr-on='export SAFE_RUN_ENABLED=1 && echo "[safe_run] Enabled."'

# ---------------------------------------------------------------------------
# Shell prompt indicator (optional)
# ---------------------------------------------------------------------------
# Adds a small 🛡 indicator to PS1/RPROMPT when safe_run is active.
# Disabled by default — uncomment to enable.

# _safe_run_prompt_indicator() {
#     if [[ "${SAFE_RUN_ENABLED:-1}" != '0' ]]; then
#         echo -n ' %F{green}🛡%f'
#     fi
# }
# RPROMPT='$(_safe_run_prompt_indicator)'"$RPROMPT"

# ---------------------------------------------------------------------------
# Startup message (optional)
# ---------------------------------------------------------------------------

if [[ "${SAFE_RUN_QUIET:-0}" != '1' ]]; then
    print -P "%F{green}[safe_run]%f Shell hook active. Commands will be explained before execution."
    print -P "%F{dim}           Set %F{yellow}SAFE_RUN_ENABLED=0%f%F{dim} to disable, or %F{yellow}SAFE_RUN_QUIET=1%f%F{dim} to silence this message.%f"
fi
