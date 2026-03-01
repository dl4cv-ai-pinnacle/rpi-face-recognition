# AGENTS.md

## Project Context
- Active implementation lives in `slop/valenia/`.
- Target device is a Raspberry Pi 5 with a 4 GiB working RAM cap for our runtime.
- Goal: build a small, reliable face-recognition pipeline that is measurable, easy to operate, and easy to extend.
- Prioritize practical offline validation, live camera usability, and reproducible metrics.

## Development Expectations
- Keep the codebase simple, modular, and easy to replace piece by piece.
- Prefer clean seams and dependency injection over hardcoded coupling.
- Follow `KISS` and `YAGNI`: do the smallest clean thing that keeps future changes straightforward.
- Favor maintainability and clarity over cleverness.

## Quality
- Use `uv` for project tooling.
- Keep typing complete and Pyright-clean.
- Keep formatting and linting clean with Ruff and pre-commit.
- Preserve reproducible benchmarks and avoid misleading metrics.
- Respect the 4 GiB RAM cap in runtime and benchmark flows unless explicitly changed.

## Branch
- `feat/#num_of_issue-description-kebab-case`
- Example: `feat/#12-add-lfw-threshold-calibration`

## Commit
- Use semantic commits: `<type>(#num_of_issue): short description`
- Example: `feat(#12): add offline lfw validation runner`

## Git
- Make small, descriptive commits at meaningful milestones.
- Do not rewrite history unless explicitly requested.
- Avoid mixing unrelated refactors, docs, and feature work in one commit when they can be separated cleanly.
