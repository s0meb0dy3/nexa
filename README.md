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

To use the Anthropic Messages protocol, add a separate profile with
`api = "anthropic"`. Keep the OpenAI profiles if you want to switch protocols:

```toml
[providers.minimax_anthropic]
api = "anthropic"
base_url = "https://api.minimax.cn/anthropic"
model = "MiniMax-M3.1-Flash-Preview"
api_key = "your-minimax-api-key"

[providers.deepseek_anthropic]
api = "anthropic"
base_url = "https://api.deepseek.com/anthropic"
model = "deepseek-flash"
api_key = "your-deepseek-api-key"
```

Select with `/model minimax_anthropic` or `/model deepseek_anthropic`.
The provider appends `/v1/messages` to `base_url`, preserves thinking signatures
and tool results, and includes cached input tokens in context usage. It reuses
the same vendor thinking levels and context limits as the OpenAI adapters.
Other Anthropic endpoints expose `default` thinking and an unknown context limit.
The current implementation supports text, thinking, and tool-use blocks; image
and redacted-thinking blocks are not supported. The output limit is 8,192 tokens
per request, including thinking.

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

Agent runs have no model-turn or tool-call count limit. They finish when the
model responds without tool calls, or stop on cancellation or failure. Bash
retains its default 30-second timeout, overridable through the tool's `timeout` argument.

## Interactive commands

Type `/` to show commands, use Up/Down to select, and Tab to complete.
Enter executes the text in the input box.

| Command | Behavior |
| --- | --- |
| `/help` | Show commands and shortcuts. |
| `/new` | Start a separate session, keeping the current model and old transcript. |
| `/tree` | Choose a complete assistant answer and continue on a branch; `/tree ID` selects it directly. Existing branches are preserved. |
| `/resume` | Search sessions for this project; `/resume ID` opens one directly. |
| `/usage` | Show API-measured context usage, the model limit, and percentage. |
| `/skill` | Search loaded skills and insert `/skill:name` into the input; add your request and press Enter to invoke. |
| `/compact` | Summarize older history and keep recent messages; `/compact FOCUS` supplies summary instructions. Escape cancels. |
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

Project instructions load automatically into the default system prompt, in order:

1. `~/.nexa/AGENTS.md`
2. `~/.agents/AGENTS.md`
3. Each `AGENTS.md` from the nearest project root down to the startup directory
4. `<cwd>/.nexa/AGENTS.md`
5. `<cwd>/.agents/AGENTS.md`

The nearest ancestor containing `.git`, `pyproject.toml`, `uv.lock`, `setup.py`,
or `package.json` is the project root. Without a marker, only the startup
directory is checked for ancestor instructions. Child directories are not scanned.
Files are concatenated with their source paths; duplicate resolved paths load
once. User-level rules apply globally, directory rules apply to that directory
and descendants, and `.nexa`/`.agents` instructions apply to their startup directory.
The prompt directs the model to follow the user's explicit request first and
more specific applicable directory rules when project rules conflict.

Instructions are read when a session is loaded, not on every API request.
Restart or resume the session after editing them. No date is added automatically,
and there is no `/reload` command or project-trust mechanism. Unreadable or invalid
UTF-8 instruction files produce an error instead of silently omitting rules.
An explicit SDK `config.system` bypasses default prompt construction.
Default coding guidelines require focused changes, preserving user edits,
project commands, relevant checks, and honest reporting. Prompt instructions
guide the model; they do not enforce filesystem permissions.

Skills load from `~/.nexa/skills/`, `~/.agents/skills/`, `<cwd>/.nexa/skills/`,
and `<cwd>/.agents/skills/`, in that order; later sources override matching names.
Use `<name>/SKILL.md` (or `<name>.md`) with a description in frontmatter:

```markdown
---
description: Review code changes and check relevant tests
---
Read the changes, identify bugs, and run the relevant tests.
```

Invoke with `/skill:review Check these changes`. Tab completes skill names.
The picker only fills the draft; it does not call the model. Unknown names show
an error. The system prompt includes skill indexes, and instructs the model to
read matching skills on demand. Explicit invocation supplies the full skill text
and its directory for resolving relative references. The `read` tool also permits
loaded skill directories, including reference files; `write` and `edit` retain
their project-only scope. This is not a sandbox for shell commands.
Restart NEXA after changing skill files; there is no hot reload.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Thinking settings are saved with the session and inherited by `/new`. Changes require an idle agent. Ctrl+T only expands/hides reasoning text. DeepSeek supports `deepseek-flash` and `deepseek-v4-pro`, with `off/low/high/max` (default: `high`). Providers without a dedicated adapter expose only `default`. Unsupported settings are rejected; switching to an incompatible model resets to the new provider’s default with a notice.

MiniMax supports `MiniMax-M3.1-Flash-Preview` with `low/medium/high/xhigh/max`
(default: `max`; thinking cannot be disabled). Profiles using `api.minimax.cn`
or `api.minimax.io` select the MiniMax adapter automatically. It preserves
`reasoning_content` in assistant history for tool continuations and reports the
official 1,000,000-token context limit. Add a profile and select it with `/model minimax`:

```toml
[providers.minimax]
base_url = "https://api.minimax.cn/v1"
model = "MiniMax-M3.1-Flash-Preview"
api_key = "your-api-key"
```

Context usage uses the latest completed API response's `total_tokens` (input history plus output), not the sum of requests. Usage is saved with assistant messages and restored with the session. Missing usage displays `—`; unknown model limits have no percentage. This snapshot excludes content added after the measured request. `/usage` shows the definition and exact token counts. After compaction, usage displays `—` until the next completed response measures the new context.

Context compaction follows Pi's summary-plus-recent-history design. `/compact`
works while idle. Automatic compaction checks before every model request,
including after a complete tool batch. It reserves the larger of 16,384 tokens
or 10% of the model window, capped at one third for small windows. It keeps about
20,000 recent tokens, capped at one fifth of a known window, without splitting
tool calls from their results. Unknown model limits permit manual compaction only.
Internal budgets combine API usage with estimates for added content; estimates
are never displayed as measured usage and cannot guarantee an exact fit.

Compaction appends a summary and retained-message boundary to the JSONL tree.
Raw history and the transcript stay intact; restarting or branching replays the
appropriate context. Repeated compaction combines the previous summary with
new older history. Failed, cancelled, empty, oversized, or ineffective summaries
leave the context unchanged; automatic failure stops the run instead of sending
an oversized request. A single oversized input or tool group may require reducing
the content rather than compaction. Summary requests use no tools.
Normal requests append to a stable prefix; summaries change only on successful
compaction. Compaction changes the message prefix, so cache reuse may decrease
temporarily; provider cache hits are not guaranteed.

Completed user, assistant, and tool messages are saved by a Harness listener before
being yielded to the UI. Cancelling a run preserves messages already completed.
A message write failure stops the run and requires reloading the saved session.
API failures are reported as runtime errors, not saved as assistant replies.
The OpenAI-compatible adapter requires both a finish reason and `[DONE]` before
committing a response; malformed stream data or tool arguments fail the run.
Previously completed messages and tool results remain available after failure.

Sessions are append-only trees. Restart restores the active root-to-node path,
including its model and thinking settings. `/tree` lists complete assistant answers across
all branches; nodes with unfinished tool calls cannot be selected. If a run was
cancelled during a tool call, return to a safe node with `/tree` before continuing.
Branching changes conversation history only; it does not undo files or commands.
