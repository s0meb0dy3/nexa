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
nexa --model deepseek-flash
```

Configure providers in `~/.nexa/config.toml`:

```toml
default_provider = "deepseek"

[providers.deepseek]
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_key = "your-api-key"
```

To install the global command from this source checkout:

```bash
uv tool install --editable . --force
```

For local development, start the interface with `uv run nexa`.
Conversations are saved per project in `~/.nexa/sessions/`.
The interface renders Markdown replies and expandable tool results. The bottom
bar shows the working directory, Git branch, provider, and active model.

- Enter sends the input; Shift+Enter or Ctrl+J inserts a newline.
- Multiline paste is supported. While a task runs, the next draft stays editable;
  it is sent only when you press Enter after the task finishes.
- Click a tool heading, or focus it and press Enter, to expand its output.
  Failures expand automatically.
- Scroll up to read without following new output; Ctrl+End returns to the latest messages.
- Escape cancels; Ctrl+T shows thinking; Ctrl+Q quits.

## Interactive commands

Type `/` to show commands, use Up/Down to select, and Tab to complete.
Enter executes the text in the input box.

| Command | Behavior |
| --- | --- |
| `/help` | Show commands and shortcuts. |
| `/new` | Start a separate session, keeping the current model and old transcript. |
| `/tree` | Choose a completed history node and continue on a branch; `/tree ID` selects it directly. Existing branches are preserved. |
| `/resume` | Search sessions for this project; `/resume ID` opens one directly. |
| `/usage` | Show API-measured context usage, the model limit, and percentage. |
| `/thinking` | Show model thinking settings; `/thinking LEVEL` changes them. DeepSeek offers `off/low/high/max` and defaults to `high`. |
| `/model` | Choose a configured provider/model; `/model PROFILE` selects a profile directly. |
| `/quit` | Cancel an active task, wait for cleanup, then exit. |

Session/model changes require an idle agent. Escape closes a selection window.
The last selected session is restored on restart, with its recorded provider and
model. Existing `default.jsonl` transcripts remain available. Session selection
is stored in the project's session directory, separately from provider config.
Unknown commands show an error; absolute paths and `/skill:name` still pass
through to the agent. Model selection uses profiles from `~/.nexa/config.toml`
and does not fetch a remote model catalog.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Thinking settings are saved with the session and inherited by `/new`. Changes require an idle agent. Ctrl+T only expands/hides reasoning text. DeepSeek supports `deepseek-flash` and `deepseek-v4-pro`, with `off/low/high/max` (default: `high`). Other providers currently expose only `default`. Unsupported settings are rejected; switching to an incompatible model resets to the new provider’s default with a notice.

Context usage uses the latest completed API response's `total_tokens` (input history plus output), not the sum of requests. Usage is saved with assistant messages and restored with the session. Missing usage displays `—`; unknown model limits have no percentage. This snapshot excludes content added after the measured request. `/usage` shows the definition and exact token counts. No token estimation or compression is performed.

Completed user, assistant, and tool messages are saved by a Harness listener before
being yielded to the UI. Cancelling a run preserves messages already completed.
A message write failure stops the run and requires reloading the saved session.

Sessions are append-only trees. Restart restores the active root-to-node path,
including its model and thinking settings. `/tree` lists safe message nodes across
all branches; nodes with unfinished tool calls cannot be selected. If a run was
cancelled during a tool call, return to a safe node with `/tree` before continuing.
Branching changes conversation history only; it does not undo files or commands.
