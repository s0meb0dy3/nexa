# NEXA

A from-scratch coding agent, following the learning guide in
[tau/dev-notes/learning-guide.md](../tau/dev-notes/learning-guide.md).

Mirrors Tau's three-layer split:

```text
nexa_coding → nexa_agent → nexa_ai
```

- `nexa_ai` — provider layer (how we talk to models).
- `nexa_agent` — the portable brain (messages, tools, loop, harness). Pure:
  no CLI, no Textual/Rich, no local config paths.
- `nexa_coding` — the coding app (CLI, file/shell tools, sessions, UI).

## Usage

NEXA only supports interactive mode. Run it from the project you want to work on:

```bash
nexa
nexa --cwd /path/to/project
nexa --provider ollama
nexa --model deepseek-chat
```

Configure providers in `~/.nexa/config.toml`:

```toml
default_provider = "deepseek"

[providers.deepseek]
base_url = "https://api.deepseek.com"
model = "deepseek-chat"
api_key = "your-api-key"
```

To install the global command from this source checkout:

```bash
uv tool install --editable . --force
```

For local development, start the interface with `uv run nexa`.
Conversations are saved per project in `~/.nexa/sessions/`.
Press Escape to cancel, Ctrl+T to show thinking, and Ctrl+Q to quit.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
