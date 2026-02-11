# Repository Guidelines

## Project Structure & Module Organization
- `server.py` houses the FastAPI app, request models, auth check, and example endpoints/data.
- `eZhire_APIs_v1.json` and `eZhire_APIs_v1.yaml` contain the OpenAPI spec references for the API surface.
- `WIDGET_AGENTS.md` documents widget authoring rules used by assistant workflows.
- `old/` and `*_agents.txt` / `*_widgets.txt` are reference notes or archives; treat as read-only unless asked.

## Build, Test, and Development Commands
- `python -m uvicorn server:app --reload` starts the API locally with hot reload (requires `fastapi` and `uvicorn`).
- `python -m pip install fastapi uvicorn` is a minimal setup if dependencies are not installed.
- No build step is defined for this repository.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 conventions.
- Prefer `snake_case` for functions/variables and `PascalCase` for Pydantic models.
- Keep FastAPI endpoints small and explicit; expand request/response models as the API grows.

## Testing Guidelines
- There are no automated tests in this repo today.
- If you add tests, place them in `tests/` and name files `test_*.py` (pytest conventions).
- Example: `pytest` runs the suite once tests are added.

## Commit & Pull Request Guidelines
- This directory is not a Git repository, so commit conventions are not discoverable.
- If you initialize Git, use clear, scoped messages (e.g., `feat: add pickup location endpoint`).
- PRs should include: summary, API changes, updates to `eZhire_APIs_v1.*`, and how the change was tested.

## Security & Configuration Tips
- `AUTH_TOKEN` in `server.py` is hard-coded; prefer loading from environment variables for real deployments.
- Avoid logging sensitive headers and tokens.

## Agent-Specific Instructions
- When producing widgets or templates, follow `WIDGET_AGENTS.md` for schema and UI constraints.
