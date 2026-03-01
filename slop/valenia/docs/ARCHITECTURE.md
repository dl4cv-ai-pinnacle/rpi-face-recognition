# Architecture Principles

This project stays intentionally small and modular to follow `KISS` + `YAGNI` while keeping room to scale.

## Design Rules

1. Keep one clear baseline path:
- `detect -> align -> embed -> compare`

2. Optimize only after measurement:
- We keep the ONNX CPU baseline first.
- Any change must be evaluated with the same validation scripts and metrics.

3. Prefer composition over framework complexity:
- `FacePipeline` composes detector + embedder.
- `LiveRuntime` owns stateful live tracking, matching, and metrics.
- Scripts orchestrate workflows (benchmark, dataset validation, HTTP serving).

4. Separate concerns:
- `src/`: reusable pipeline logic.
- `scripts/`: executable workflows.
- `docs/metrics/`: reproducible benchmark outputs.

## Extension Points

- Swap detector model:
  - Change `--det-model` in scripts.
  - Keep `SCRFDDetector` interface stable (`detect(...) -> boxes + landmarks`).

- Swap embedding model:
  - Change `--rec-model`.
  - Keep `ArcFaceEmbedder.get_embedding(...)` contract stable.

- Add tracker / identity memory:
  - Extend `LiveRuntime` for stateful live behavior.
  - Keep `live_camera_server.py` focused on camera + HTTP transport.
  - Keep validation script unchanged for regression tracking.

## What We Explicitly Avoid (for now)

- No premature plugin system.
- No database abstraction layer yet.
- No multiple runtime backends in one code path until baseline metrics require it.

## Quality Gate

Use `uv` tooling only:

```bash
uv run --group dev pre-commit run --all-files
```

This runs:
- `ruff format --check`
- `ruff check --fix`
- `pyright`
