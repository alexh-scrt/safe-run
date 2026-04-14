# safe_run 🛡️

**AI-powered shell command wrapper** — explains what any command does in plain English before executing it, with color-coded risk assessment and confirmation prompts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

`safe_run` wraps any shell command through a three-step flow:

1. **Explain** — Sends the command to an LLM (OpenAI GPT-4o or a local Ollama model) which returns a plain-English explanation and structured risk assessment.
2. **Assess** — A built-in rule engine pre-screens 30+ dangerous patterns (`rm -rf`, `curl | bash`, `chmod 777`, etc.) and assigns a **LOW / MEDIUM / HIGH / CRITICAL** risk badge before any LLM call is made.
3. **Confirm** — Shows a Rich-formatted panel and prompts you to approve. Low-risk commands can be auto-approved; CRITICAL commands require you to type `yes` in full.

```
╭─ safe_run 💀 CRITICAL ──────────────────────────────────────────────╮
│                                                                      │
│  Command: rm -rf /                                                   │
│                                                                      │
│  What it does:                                                       │
│  Recursively and forcibly deletes every file and directory on the    │
│  filesystem root. This will destroy your entire operating system     │
│  and all data on the disk.                                           │
│                                                                      │
│  Risk: 💀 CRITICAL  (score: 100/100)                                │
│                                                                      │
│  Risk factors:                                                       │
│    • Recursive forced deletion from filesystem root                  │
│    • Recursive forced file/directory deletion (rm -rf)               │
│                                                                      │
│  Effects:                                                            │
│    • Deletes all files on the system                                 │
│    • Renders system unbootable                                       │
│                                                                      │
│  Reversible: No                                                      │
│                                                                      │
│  Explained by: openai/gpt-4o                                         │
╰──────────────────────────────────────────────────────────────────────╯

╔══════════════════════════════════════════════════════════════════════╗
║                    💀 CRITICAL                                       ║
║  ⚠  CRITICAL RISK COMMAND                                           ║
║                                                                      ║
║  This command may cause irreversible damage to your system,          ║
║  data, or security. It cannot be undone.                             ║
║                                                                      ║
║  If you are absolutely certain, type yes (exactly, lowercase)        ║
║  to continue. Any other input will abort.                            ║
╚══════════════════════════════════════════════════════════════════════╝

Type yes to confirm, anything else to abort:
```

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic usage](#basic-usage)
  - [Flags and options](#flags-and-options)
  - [Sub-commands](#sub-commands)
- [Shell Integration](#shell-integration)
  - [Zsh (.zshrc)](#zsh-zshrc)
  - [Bash (.bashrc)](#bash-bashrc)
  - [Shell integration options](#shell-integration-options)
- [Configuration](#configuration)
  - [File location](#file-location)
  - [Generating a default config](#generating-a-default-config)
  - [Full configuration reference](#full-configuration-reference)
  - [Using Ollama (local LLM)](#using-ollama-local-llm)
- [Risk Levels](#risk-levels)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

---

## Features

- 🔍 **Plain-English LLM explanations** — Supports OpenAI GPT-4o and any local [Ollama](https://ollama.ai) model via its OpenAI-compatible API.
- 🚨 **30+ built-in risk rules** — Detects `rm -rf`, `curl | bash`, `mkfs`, `chmod 777`, firewall changes, reverse shells, fork bombs, and more — *before* the LLM call.
- 🎨 **Color-coded risk badges** — `LOW` (green) / `MEDIUM` (yellow) / `HIGH` (orange) / `CRITICAL` (red) with score out of 100.
- ✅ **Smart confirmation prompts** — Auto-approves low-risk commands, standard yes/no for HIGH, and requires typing `yes` for CRITICAL.
- ⚡ **Zero-friction shell integration** — One `source` line in `.zshrc` or `.bashrc`.
- 🔧 **Configurable** — TOML config file for provider, model, thresholds, allowlists, and blocklists.
- 🚀 **CI/script friendly** — `--bypass` flag skips all checks for automation.
- 🔌 **Graceful fallback** — If OpenAI fails, automatically tries Ollama (and vice versa).

---

## Requirements

- Python **3.9** or newer
- An **OpenAI API key** (for the default OpenAI provider), **or** a running [Ollama](https://ollama.ai) instance (for local/offline use)
- A POSIX-compatible shell (Linux, macOS). Windows is not officially supported.

---

## Installation

### From PyPI (recommended)

```bash
pip install safe_run
```

### From source

```bash
git clone https://github.com/safe_run/safe_run.git
cd safe_run
pip install -e .
```

### Verify installation

```bash
safe_run --version
# safe_run, version 0.1.0

safe_run --help
```

---

## Quick Start

1. **Set your OpenAI API key:**

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   Or store it in the [config file](#configuration).

2. **Run any command through safe_run:**

   ```bash
   safe_run ls -la /tmp
   safe_run -- rm -rf /tmp/old_project
   safe_run -- curl https://example.com/install.sh | bash
   ```

   The `--` separator is recommended when your command contains flags that might be misinterpreted by safe_run itself.

3. **Add shell integration** (optional but recommended — see [Shell Integration](#shell-integration)).

---

## Usage

### Basic usage

```bash
# Explain and confirm before running
safe_run <command> [args...]

# Use -- to clearly separate safe_run flags from the command
safe_run [safe_run-flags] -- <command> [args...]
```

**Examples:**

```bash
safe_run echo "hello world"
safe_run -- git push origin main --force
safe_run -- docker system prune -af
safe_run -- chmod 777 /var/www/html
```

### Flags and options

| Flag | Short | Description |
|------|-------|-------------|
| `--bypass` | | Execute immediately without explanation or confirmation. For scripts and CI. |
| `--no-explain` | | Skip the LLM explanation; only run rule-based pre-screening. Confirmation prompt still shown if above threshold. |
| `--yes` | `-y` | Auto-confirm commands below CRITICAL level. CRITICAL always requires typed confirmation. |
| `--verbose` | `-v` | Show stdout/stderr in the result summary even on success. |
| `--provider` | | Override the LLM provider for this run: `openai` or `ollama`. |
| `--config` | | Path to a custom TOML config file. |
| `--no-color` | | Disable Rich color output (useful for log files). |
| `--version` | | Show the version and exit. |
| `--help` | `-h` | Show help message and exit. |

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (alternative to config file). |
| `SAFE_RUN_CONFIG` | Path to the TOML config file (overrides default location). |

### Sub-commands

#### `safe_run check` — Risk-check without executing

Runs only the rule-based pre-screener (no LLM call, no execution). Exits with code 1 if the command is HIGH or CRITICAL.

```bash
safe_run check rm -rf /tmp/old
safe_run check -- curl https://example.com | bash
safe_run check chmod 777 /var/www
```

Useful in CI pipelines to gate on dangerous commands:

```bash
# In a CI script — fail if the command is dangerous
if ! safe_run check -- $MY_COMMAND; then
  echo "Dangerous command detected — aborting."
  exit 1
fi
```

#### `safe_run init-config` — Generate a default config file

Writes a commented default configuration to `~/.config/safe_run/config.toml` (or a custom path). Does **not** overwrite an existing file.

```bash
safe_run init-config
safe_run init-config --path /path/to/my/config.toml
```

---

## Shell Integration

Shell integration wraps every interactive command you type with `safe_run` automatically, so you get explanations and risk assessment for *all* commands — not just ones you prefix with `safe_run`.

### Zsh (.zshrc)

Add the following to your `~/.zshrc`:

```zsh
# safe_run shell integration
source "$(python3 -m site --user-base)/share/safe_run/safe_run.zsh"
```

Or, if you cloned the repository:

```zsh
source /path/to/safe_run/shell/safe_run.zsh
```

Then reload your shell:

```bash
source ~/.zshrc
```

### Bash (.bashrc)

Add the following to your `~/.bashrc`:

```bash
# safe_run shell integration
source "$(python3 -m site --user-base)/share/safe_run/safe_run.bash"
```

Or, if you cloned the repository:

```bash
source /path/to/safe_run/shell/safe_run.bash
```

Then reload your shell:

```bash
source ~/.bashrc
```

> **Note on bash-preexec:** For best results in Bash, install [bash-preexec](https://github.com/rcaloras/bash-preexec) before sourcing `safe_run.bash`. Without it, the hook falls back to a `DEBUG` trap which has some limitations with complex pipelines.

### Shell integration options

Configure the shell hook via environment variables in your `.zshrc` / `.bashrc`:

```bash
# Disable safe_run for the current session (without removing the hook)
export SAFE_RUN_ENABLED=0

# Re-enable it
export SAFE_RUN_ENABLED=1

# Use Ollama instead of OpenAI for all commands
export SAFE_RUN_OPTS='--provider ollama'

# Silence the startup banner
export SAFE_RUN_QUIET=1

# Add extra bypass prefixes (space-separated)
# Commands starting with these prefixes skip safe_run entirely
export SAFE_RUN_BYPASS_COMMANDS+=' man info htop'

# Specify a custom safe_run binary path
export SAFE_RUN_BIN='/custom/path/to/safe_run'
```

**Convenience aliases** added by the shell snippets:

| Alias | Description |
|-------|-------------|
| `sr` | Shorthand for `safe_run` |
| `srb` | `safe_run --bypass --` (bypass mode) |
| `src` | `safe_run check` (risk check only) |
| `sr-off` | Disable safe_run for the current session |
| `sr-on` | Re-enable safe_run for the current session |

**How the shell hook works:**

The Zsh hook uses `preexec` (via `add-zsh-hook`). The Bash hook uses `bash-preexec` if available, or falls back to a `DEBUG` trap. Both hooks:

1. Intercept every command before it runs.
2. Pass it to `safe_run` for explanation and confirmation.
3. If approved, `safe_run` executes the command internally via subprocess (preventing double execution via `SIGINT` to cancel the shell's own pending execution).
4. If declined or blocked, the original command is cancelled.

> **Bypass list:** Read-only commands like `ls`, `cat`, `echo`, `grep`, `pwd`, `cd`, and shell built-ins are automatically bypassed so you don't get explanations for every trivial operation.

---

## Configuration

### File location

By default, safe_run looks for its configuration at:

```
~/.config/safe_run/config.toml
```

This can be overridden with:

```bash
# Environment variable
export SAFE_RUN_CONFIG=/path/to/my/config.toml

# Or CLI flag
safe_run --config /path/to/my/config.toml -- <command>
```

### Generating a default config

```bash
safe_run init-config
```

This creates a fully commented `~/.config/safe_run/config.toml` with all options documented. It will not overwrite an existing file.

### Full configuration reference

```toml
# ~/.config/safe_run/config.toml

# ── LLM Provider ────────────────────────────────────────────────────────────
[llm]
# LLM backend to use: "openai" or "ollama"
provider = "openai"

# OpenAI model name (used when provider = "openai")
# Recommended: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
openai_model = "gpt-4o"

# OpenAI API key.
# If omitted, falls back to the OPENAI_API_KEY environment variable.
# openai_api_key = "sk-..."

# Ollama model name (used when provider = "ollama")
# Examples: "llama3", "mistral", "codellama", "phi3"
ollama_model = "llama3"

# Ollama OpenAI-compatible API base URL
ollama_base_url = "http://localhost:11434/v1"

# HTTP request timeout in seconds for LLM API calls
timeout = 30

# ── Risk Assessment ─────────────────────────────────────────────────────────
[risk]
# Minimum risk level requiring explicit user confirmation.
# One of: LOW, MEDIUM, HIGH, CRITICAL
# - LOW:      Confirm everything (maximum safety)
# - MEDIUM:   Confirm MEDIUM and above
# - HIGH:     Confirm HIGH and CRITICAL (recommended default)
# - CRITICAL: Only confirm truly catastrophic commands
threshold = "HIGH"

# If true, commands BELOW the threshold are auto-approved without prompting.
auto_confirm_below_threshold = true

# Commands that always pass without LLM explanation or confirmation.
# Matched as a prefix of the stripped command string.
# Example: allowlist = ["echo", "ls", "git status"]
allowlist = []

# Commands that are always blocked and never executed.
# Matched as a prefix of the stripped command string.
# Example: blocklist = ["rm -rf /", "mkfs"]
blocklist = []

# ── Command Handling ─────────────────────────────────────────────────────────
[commands]
# Maximum number of characters to send to the LLM.
# Longer commands are truncated before submission.
max_command_length = 2000

# ── Display ──────────────────────────────────────────────────────────────────
[display]
# Show the raw command string in the explanation panel header.
show_raw_command = true

# ── Logging ──────────────────────────────────────────────────────────────────
[logging]
# Logging verbosity for safe_run's own output.
# One of: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = "WARNING"
```

### Using Ollama (local LLM)

[Ollama](https://ollama.ai) lets you run LLMs locally without an internet connection or API key.

**1. Install Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

**2. Pull a model:**

```bash
ollama pull llama3        # Meta Llama 3 8B (recommended)
ollama pull mistral       # Mistral 7B
ollama pull phi3          # Microsoft Phi-3 (smaller, faster)
ollama pull codellama     # Code-optimized
```

**3. Start Ollama:**

```bash
ollama serve
```

**4. Configure safe_run to use Ollama:**

```toml
# ~/.config/safe_run/config.toml
[llm]
provider = "ollama"
ollama_model = "llama3"
ollama_base_url = "http://localhost:11434/v1"
timeout = 60   # Local models may be slower
```

Or use the CLI flag for a one-off run:

```bash
safe_run --provider ollama -- rm -rf /tmp/old_project
```

**Automatic fallback:** If your primary provider fails (e.g. no internet, expired API key), safe_run automatically tries the other provider. If both fail, it falls back to rule-based assessment only and elevates the risk to HIGH.

---

## Risk Levels

Every command receives one of four risk levels, derived from the *higher* of the rule-based pre-screener result and the LLM's assessment:

| Level | Badge | Color | Score | Examples | Confirmation |
|-------|-------|-------|-------|----------|--------------|
| **LOW** | ✅ LOW | Green | 0–29 | `ls`, `echo`, `cat`, `grep`, `pip install`, `wget` (download only) | Auto-approved (configurable) |
| **MEDIUM** | ⚠️ MEDIUM | Yellow | 30–59 | `git reset --hard`, `curl -k`, `eval`, `chmod -R`, `nmap`, `rsync --delete` | Standard yes/no prompt |
| **HIGH** | 🚨 HIGH | Orange | 60–89 | `sudo`, `chmod 777`, `iptables -F`, `shutdown`, `DROP DATABASE`, `killall` | Standard yes/no prompt (always shown) |
| **CRITICAL** | 💀 CRITICAL | Red | 90–100 | `rm -rf /`, `curl \| bash`, `mkfs`, `dd` to disk, fork bomb, reverse shell | Must type `yes` exactly |

### Built-in risk rules (30+ patterns)

The rule engine detects:

- **File destruction:** `rm -rf /`, `rm -rf` on system directories (`/etc`, `/usr`, `/bin`, etc.)
- **Remote code execution:** `curl | bash`, `wget | sh`, `base64 -d | bash`
- **Disk operations:** `dd of=/dev/sdX`, `mkfs.*`, `> /dev/sdX`
- **Privilege escalation:** `sudo`, `su root`, setuid bits (`chmod u+s`)
- **Firewall changes:** `iptables -F`, `iptables -P INPUT ACCEPT`, `ufw disable`
- **Reverse shells:** `nc -e /bin/bash`, Python socket reverse shells
- **Fork bombs:** `: () { :|: & }; :`
- **Credential exposure:** Passwords/tokens in command-line arguments
- **SSH backdoors:** Writing to `~/.ssh/authorized_keys`
- **Cron/sudoers modification:** `crontab -e`, writing to `/etc/sudoers`
- **System shutdown:** `shutdown`, `reboot`, `halt`, `poweroff`
- **Database destruction:** `DROP DATABASE`, `DROP TABLE`
- **Dangerous git operations:** `git push --force`, `git reset --hard`, `git clean -f`
- **Network scanning:** `nmap`, `masscan`, `zmap`
- **Insecure downloads:** `curl -k`, `wget --no-check-certificate`
- **Anti-forensics:** `history -c`, overwriting `~/.bash_history`
- **Many more...**

---

## Examples

### Explain a safe command

```bash
$ safe_run ls -la /tmp

╭─ safe_run ✅ LOW ──────────────────────────────────────────────────╮
│  Command: ls -la /tmp                                               │
│                                                                     │
│  What it does:                                                      │
│  Lists all files (including hidden ones) in the /tmp directory      │
│  with detailed information including permissions, owner, size,      │
│  and modification time.                                             │
│                                                                     │
│  Risk: ✅ LOW  (score: 0/100)                                      │
│  Reversible: Yes                                                    │
│  Explained by: openai/gpt-4o                                        │
╰─────────────────────────────────────────────────────────────────────╯
✓ Auto-confirmed ✅ LOW command.
[running ls -la /tmp ...]
```

### Explain a HIGH-risk command

```bash
$ safe_run -- chmod 777 /var/www/html

╭─ safe_run 🚨 HIGH ─────────────────────────────────────────────────╮
│  Command: chmod 777 /var/www/html                                   │
│                                                                     │
│  What it does:                                                      │
│  Sets world-readable, world-writable, and world-executable          │
│  permissions on /var/www/html, meaning any user on the system       │
│  can read, modify, or execute files in this directory.              │
│                                                                     │
│  Risk: 🚨 HIGH  (score: 60/100)                                    │
│                                                                     │
│  Risk factors:                                                      │
│    • Sets world-readable/writable/executable permissions (chmod 777)│
│    • Severe security risk on a web-accessible directory             │
│                                                                     │
│  Effects:                                                           │
│    • All users gain full access to web root                         │
│    • Potential for arbitrary file upload attacks                    │
│                                                                     │
│  Reversible: Yes                                                    │
│  Explained by: openai/gpt-4o                                        │
╰─────────────────────────────────────────────────────────────────────╯

Run this 🚨 HIGH command? [y/N] n

Command not executed.
```

### Check risk without executing

```bash
$ safe_run check -- curl https://example.com/install.sh | bash
💀 CRITICAL  score=100/100

  • Downloads and pipes remote code directly into a shell interpreter
  • Arbitrary code execution risk

$ echo $?
1
```

### Use Ollama for a single run

```bash
$ safe_run --provider ollama -- git push origin main --force
```

### Bypass mode for CI scripts

```bash
#!/bin/bash
# deploy.sh — run commands without safe_run prompts
set -euo pipefail

safe_run --bypass -- docker build -t myapp:latest .
safe_run --bypass -- docker push myapp:latest
safe_run --bypass -- kubectl rollout restart deployment/myapp
```

### Auto-confirm below HIGH

```bash
# -y auto-approves LOW/MEDIUM; HIGH still prompts; CRITICAL still requires 'yes'
safe_run -y -- git clean -fd
```

### Configure an allowlist for common safe commands

```toml
# ~/.config/safe_run/config.toml
[risk]
allowlist = [
  "echo",
  "ls",
  "cat",
  "git status",
  "git log",
  "git diff",
  "docker ps",
  "kubectl get",
]
```

### Configure a blocklist for commands to always block

```toml
# ~/.config/safe_run/config.toml
[risk]
blocklist = [
  "rm -rf /",
  "mkfs",
  "dd if=/dev/zero",
]
```

---

## Development

### Setup

```bash
git clone https://github.com/safe_run/safe_run.git
cd safe_run

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e '.[dev]'
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=safe_run --cov-report=term-missing

# Run a specific test module
pytest tests/test_risk.py -v

# Run a specific test
pytest tests/test_risk.py::TestAssessRiskCriticalCommands::test_curl_pipe_bash -v
```

### Project structure

```
safe_run/
├── safe_run/
│   ├── __init__.py       # Package init, version
│   ├── main.py           # Click CLI entry point and orchestration
│   ├── config.py         # TOML config loader and SafeRunConfig dataclass
│   ├── risk.py           # Rule-based risk pre-screener (30+ patterns)
│   ├── explainer.py      # LLM client (OpenAI + Ollama)
│   ├── executor.py       # Subprocess command executor
│   └── display.py        # Rich terminal UI components
├── tests/
│   ├── test_config.py
│   ├── test_risk.py
│   ├── test_explainer.py
│   └── test_executor.py
├── shell/
│   ├── safe_run.zsh      # Zsh shell hook
│   └── safe_run.bash     # Bash shell hook
├── pyproject.toml
└── README.md
```

### Architecture notes

**Data flow:**

```
CLI args
  └─► config.py (load config)
  └─► risk.py (rule-based pre-screening)
  └─► explainer.py (LLM explanation + risk)
  └─► display.py (Rich UI: panel + prompt)
  └─► executor.py (subprocess execution)
  └─► display.py (result summary)
```

**Risk level resolution:** The final risk level is `max(rule_level, llm_level)` — neither source can downgrade a risk flagged by the other.

**LLM fallback chain:** `primary_provider → fallback_provider → degraded_result(HIGH risk)`

### Adding new risk rules

Risk rules live in `safe_run/risk.py` in the `RISK_RULES` list. Each rule is a `RiskRule` named tuple with:

```python
_rule(
    name="my_new_rule",          # Unique snake_case identifier
    regex=r"\bmy_pattern\b",     # Regular expression (re.IGNORECASE by default)
    level=RiskLevel.HIGH,         # LOW / MEDIUM / HIGH / CRITICAL
    reason="Why this is risky.", # Human-readable explanation
)
```

Add your rule to the `RISK_RULES` list and add a corresponding test in `tests/test_risk.py`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Add tests for any new functionality or bug fixes.
3. Ensure all tests pass: `pytest`
4. Open a pull request with a clear description of the change.

**Reporting issues:** Please open a GitHub issue with the command that triggered unexpected behavior and your `safe_run --version` output.

---

## Acknowledgements

- [Rich](https://github.com/Textualize/rich) — Beautiful terminal formatting
- [Click](https://click.palletsprojects.com/) — CLI framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM client
- [Ollama](https://ollama.ai) — Local LLM runtime
- [bash-preexec](https://github.com/rcaloras/bash-preexec) — Bash preexec hook (optional)
