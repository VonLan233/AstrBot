# AstrBot Agent Guide

This repository already uses `AGENTS.md` as the authoritative instruction file.
Use this file as a quick project map before making changes.

## Project layout

- `main.py`: local application entry point.
- `astrbot/`: Python backend package.
  - `astrbot/core/`: core runtime, provider management, agents, sessions, config, and utilities.
  - `astrbot/api/`: backend API surface.
  - `astrbot/builtin_stars/`: built-in plugin implementations.
  - `astrbot/cli/`: command-line entry points.
- `dashboard/`: Vue/Vite WebUI project.
- `tests/`: pytest suite with unit, agent, and fixture directories.
- `docs/`, `samples/`, `scripts/`, `k8s/`, `openspec/`: documentation, examples, automation, deployment, and specs.

## Common commands

```bash
uv sync
uv run main.py
uv run ruff format .
uv run ruff check .
```

For WebUI work:

```bash
cd dashboard
pnpm install
pnpm dev
```

## Development notes

- Follow `AGENTS.md` for setup, linting, PR, and repository-specific rules.
- Keep backend changes in `astrbot/` and WebUI changes in `dashboard/` unless a feature explicitly crosses both layers.
- Use `pathlib.Path` and AstrBot path utilities for filesystem paths.
- Keep comments and PR descriptions in English.
- Do not add generated report files such as `*_SUMMARY.md`.
