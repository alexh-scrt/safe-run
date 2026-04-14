# safe_run 🛡️

**Know before you run** — AI-powered shell command wrapper that explains what any command does in plain English, flags dangerous operations with color-coded risk levels, and requires confirmation before executing anything risky.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

`safe_run` wraps any shell command through a three-step flow: it sends the command to an LLM (OpenAI GPT-4o or a local Ollama model) for a plain-English explanation, runs it through a rule-based engine that detects 30+ dangerous patterns and assigns a **LOW / MEDIUM / HIGH / CRITICAL** risk badge, then prompts you for confirmation before executing. Low-risk commands can be auto-approved, while CRITICAL commands require you to type `yes` in full. Integrate it as a shell function in zsh or bash and get AI-powered transparency for every command you run — without changing your workflow.

---

## Quick Start

**Install:**

```bash
pip install safe_run
```

**Run a command through safe_run:**

```bash
# Wrap any command — safe_run explains and confirms before executing
safe_run ls -la /tmp
safe_run -- rm -rf /tmp/old_project
safe_run -- curl https://example.com | bash
```

**Integrate into your shell (optional but recommended):**

```bash
# Zsh
echo 'source "$(python3 -m site --user-base)/share/safe_run/safe_run.zsh"' >> ~/.zshrc
source ~/.zshrc

# Bash
echo 'source "$(python3 -m site --user-base)/share/safe_run/safe_run.bash"' >> ~/.bashrc
source ~/.bashrc
```

Once sourced, every interactive command you type is automatically wrapped — no extra typing required.

---

## Features

- **Plain-English LLM explanations** — Supports OpenAI GPT-4o and local Ollama models; falls back gracefully if the primary provider is unavailable.
- **Rule-based pre-screener** — Catches 30+ dangerous patterns (`rm -rf`, `chmod 777`, `curl | bash`, `dd if=`, `:(){ :|:& };:`, and more) before making any LLM call.
- **Color-coded risk badges** — Commands are labeled LOW (green), MEDIUM (yellow), HIGH (orange), or CRITICAL (red) with a matching Rich-formatted panel.
- **Smart confirmation prompts** — Auto-approves low-risk commands, requires a single keypress for medium/high, and demands you type `yes` in full for CRITICAL operations.
- **Zero-friction shell integration** — A single `source` line in `.zshrc` or `.bashrc`; use `--bypass` to skip safe_run in scripts and CI pipelines.

---

## Usage Examples

### Basic command wrapping

```bash
# safe_run explains the command, shows risk level, then asks to confirm
$ safe_run git push origin main --force

╭─ Command ────────────────────────────────────────────────────────────╮
│  git push origin main --force                                        │
╰──────────────────────────────────────────────────────────────────────╯
╭─ Explanation  ⚠ HIGH ────────────────────────────────────────────────╮
│  Force-pushes the local 'main' branch to the 'origin' remote,        │
│  overwriting the remote history. This can permanently discard         │
│  commits that others may have already pulled.                         │
│                                                                       │
│  Risk factors: destructive remote history rewrite                     │
╰──────────────────────────────────────────────────────────────────────╯
Proceed? [y/N]:
```

### CRITICAL command — requires full confirmation

```bash
$ safe_run -- rm -rf /

╭─ Command ────────────────────────────────────────────────────────────╮
│  rm -rf /                                                            │
╰──────────────────────────────────────────────────────────────────────╯
╭─ Explanation  💀 CRITICAL ────────────────────────────────────────────╮
│  Recursively and forcefully deletes every file on the system         │
│  starting from the root directory. This will destroy the OS and      │
│  all data. This operation is irreversible.                            │
│                                                                       │
│  Risk factors: recursive forced deletion from root                    │
╰──────────────────────────────────────────────────────────────────────╯
This command is CRITICAL. Type 'yes' to confirm or anything else to abort: _
```

### Skip explanation in scripts or CI

```bash
# --bypass executes immediately without explanation or confirmation
safe_run --bypass -- npm install

# --no-explain skips the LLM call but still shows the risk badge
safe_run --no-explain -- docker system prune -af
```

### Using a local Ollama model

```bash
# Override provider at runtime
SAFE_RUN_LLM_PROVIDER=ollama safe_run -- kubectl delete namespace production
```

---

## Project Structure

```
safe_run/
├── safe_run/
│   ├── __init__.py        # Package init, version, entry point
│   ├── main.py            # Click CLI — orchestrates explain-confirm-execute
│   ├── explainer.py       # LLM client (OpenAI / Ollama) with fallback logic
│   ├── risk.py            # Rule-based pre-screener for 30+ dangerous patterns
│   ├── executor.py        # subprocess runner with streaming output & timeouts
│   ├── display.py         # Rich terminal UI — panels, badges, prompts
│   └── config.py          # Config loader for ~/.config/safe_run/config.toml
├── tests/
│   ├── test_risk.py       # Unit tests for risk detection and severity scoring
│   ├── test_explainer.py  # Unit tests for LLM client with mocked responses
│   ├── test_executor.py   # Unit tests for execution, timeouts, output capture
│   └── test_config.py     # Unit tests for config loading and validation
├── shell/
│   ├── safe_run.zsh       # Zsh shell integration snippet
│   └── safe_run.bash      # Bash shell integration snippet
├── pyproject.toml
└── README.md
```

---

## Configuration

On first run, `safe_run` creates a default config at `~/.config/safe_run/config.toml`. Edit it to customize behavior:

```toml
[llm]
provider = "openai"          # "openai" or "ollama"
model = "gpt-4o"             # e.g. "gpt-4o", "llama3", "mistral"
ollama_base_url = "http://localhost:11434"

[risk]
threshold = "HIGH"           # Auto-confirm below this level: LOW, MEDIUM, HIGH, CRITICAL
auto_confirm_low = true      # Auto-approve LOW risk commands without prompting

[allowlist]
# Commands that always bypass safe_run (no explanation, no confirmation)
commands = [
    "ls",
    "cat",
    "echo",
    "pwd",
    "git status",
    "git log",
]

[blocklist]
# Commands that are always hard-blocked and can never be executed
commands = [
    "rm -rf /",
    "mkfs",
]

[execution]
timeout = 300                # Command timeout in seconds (0 = no limit)
```

**Environment variable overrides:**

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `SAFE_RUN_LLM_PROVIDER` | Override LLM provider (`openai` or `ollama`) |
| `SAFE_RUN_MODEL` | Override the model name |
| `SAFE_RUN_BYPASS` | Set to `1` to bypass safe_run entirely (useful in CI) |
| `SAFE_RUN_CONFIG` | Path to an alternate config file |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

> Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.
