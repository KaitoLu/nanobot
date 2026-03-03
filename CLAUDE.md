# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (development):**
```bash
pip install -e ".[dev]"        # Standard
uv pip install -e ".[dev]"     # With uv (recommended)
pip install -e ".[dev,matrix]" # Include Matrix channel support
```

**Test:**
```bash
pytest tests/                        # All tests
pytest tests/test_foo.py             # Single file
pytest tests/test_foo.py::test_bar   # Single test
```

**Lint:**
```bash
ruff check .          # Check
ruff check . --fix    # Auto-fix
ruff format .         # Format
```

**Docker:**
```bash
# Build and run via Docker Compose (mounts ~/.nanobot)
docker compose run --rm nanobot-cli onboard   # Initialize config
docker compose run --rm nanobot-cli status    # Show status
docker compose run --rm nanobot-cli agent     # Interactive chat
docker compose up -d nanobot-gateway          # Start gateway (port 18790)
```

**CLI (after install):**
```bash
nanobot onboard          # First-time setup: creates ~/.nanobot/config.json
nanobot status           # Show configured providers/channels
nanobot agent            # Interactive agent session
nanobot agent -m "msg"   # Single-turn message
nanobot gateway          # Start chat channel gateway
nanobot cron list        # List scheduled tasks
nanobot cron add "..."   # Add cron job
```

## Architecture

Nanobot is a lightweight personal AI assistant framework (~4K lines of core code). It has four layers:

```
Chat Channels  →  Message Bus  →  Agent Loop  →  LLM Providers
(Telegram, Discord, WhatsApp, Feishu, Slack, Email, QQ, Matrix, …)
```

### Message Bus (`nanobot/bus/`)
Loose coupling between channels and agent via two async queues:
- `InboundMessage`: channel → agent (includes sender, chat_id, content, media, metadata)
- `OutboundMessage`: agent → channel (includes chat_id, content, reply_to, media)

### Agent Loop (`nanobot/agent/loop.py`)
Core execution: receives `InboundMessage`, builds context, calls LLM, executes tool calls, loops until done (max 40 iterations), sends `OutboundMessage`. Also handles session consolidation to memory files and subagent coordination.

### Context Builder (`nanobot/agent/context.py`)
Assembles the system prompt from: identity/runtime info → bootstrap markdown files (AGENTS.md, SOUL.md, USER.md, TOOLS.md, IDENTITY.md from workspace) → MEMORY.md content → always-active skills → skills index.

### Provider Layer (`nanobot/providers/`)
- `registry.py`: metadata for 37+ providers (no if-elif chains); auto-detects gateways by API key prefix
- `litellm_provider.py`: LiteLLM-based wrapper (most providers)
- `custom_provider.py`: direct OpenAI-compatible endpoint

### Channel System (`nanobot/channels/`)
12 channel implementations all extending `BaseChannel` (`start`, `stop`, `handle_message`, `send_message`). WhatsApp uses an external Node.js bridge (`bridge/`) communicating via WebSocket (Baileys library).

### Session & Memory (`nanobot/session/`, `nanobot/agent/memory.py`)
Sessions stored as JSONL (append-only, keyed by `channel:chat_id`). Periodic consolidation writes summaries to `MEMORY.md` and timestamped entries to `HISTORY.md` in the workspace.

### Skills System (`nanobot/agent/skills.py`)
Skills are markdown files (`SKILL.md`) the LLM can request to load. Located in:
- Builtin: `nanobot/skills/{name}/SKILL.md`
- User workspace: `~/.nanobot/workspace/skills/{name}/SKILL.md`

### Tools (`nanobot/agent/tools/`)
9 built-in tools: Shell, Filesystem (read/write/edit/list), Web (search/fetch), Message (cross-channel), Cron, Spawn (subagents), MCP (dynamic external tools).

### Config (`nanobot/config/schema.py`)
Pydantic BaseSettings with camelCase/snake_case alias support. Top-level keys: `providers`, `agents`, `channels`, `tools`, `security`. Config file: `~/.nanobot/config.json`.

### Cron & Heartbeat (`nanobot/cron/`, `nanobot/heartbeat/`)
Cron jobs use croniter for scheduling. Heartbeat service reads `HEARTBEAT.md` from workspace and fires virtual tool-call events to wake the agent on a schedule.

## Key Design Decisions

- **Provider registry over if-elif**: `registry.py` is the single source of truth for all provider metadata.
- **Append-only sessions**: JSONL format preserves LLM prompt cache efficiency.
- **Bootstrap files**: Behavior is customized via markdown files (SOUL.md, USER.md, etc.) without code changes.
- **Workspace sandbox**: File/shell tools can optionally be restricted to `~/.nanobot/workspace/`.
- **Deny-by-default channels**: Empty `allowFrom` list in channel config rejects all senders; use `["*"]` to allow all.
- **Matrix E2EE**: Matrix channel requires the optional `matrix` extra (`pip install -e ".[matrix]"`).
