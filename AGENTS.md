# AGENTS.md

## Project

Modular face-recognition pipeline for Raspberry Pi 5 (4 GiB RAM cap). Swappable backends behind `@runtime_checkable` Protocols, YAML config, FAISS matching, live MJPEG dashboard.

## Architecture (quick)

```
Camera → Detection → Tracking → Alignment → Embedding → Matching (FAISS) → Gallery
```

Swappable: detection (insightface/ultraface), alignment (cv2/skimage), embedding INT8 toggle. See `docs/ARCHITECTURE.md` for full details.

## Key files

| Path | What |
|---|---|
| `src/contracts.py` | All Protocols + type aliases — read this first |
| `src/config.py` | YAML → frozen dataclass tree |
| `src/pipeline.py` | FacePipeline + `build_pipeline()` factory |
| `src/gallery.py` | GalleryStore + FAISS index + unknowns workflow |
| `src/live.py` | LiveRuntime orchestrator + `swap_pipeline()` |
| `config.yaml` | All defaults — detection, alignment, matching, tracking, etc. |
| `server/app.py` | HTTP server entry point |

## Development

- **Tooling:** `uv` for everything. Python ≥3.13. `uv run --python 3.13 pytest`, `uv run --python 3.13 pyright src/`, `uv run --python 3.13 ruff check src/`.
- **Pi OS deps:** when a task needs Raspberry Pi camera or system libraries, install the required OS-level packages up front with `apt` rather than deferring them to the user. This includes things like `python3-picamera2`, `python3-opencv`, `libcap-dev`, and `libcamera` bindings when the setup path depends on them.
- **uv + Pi bindings:** if the runtime needs `apt`-installed camera bindings, create the project venv with `uv venv --python 3.13 --system-site-packages` before `uv sync` so `uv run` can import those system modules.
- **Service management:** on Raspberry Pi, manage the live app through `bash scripts/service.sh ...` instead of ad-hoc `systemctl` or manual background commands whenever that script covers the task.
- **Typing:** Pyright strict mode. All public functions annotated.
- **Linting:** Ruff with `E, F, I, UP, B, SIM` rules.
- **Tests:** `tests/` — behavioral tests with DI stubs, no real ONNX models needed.
- **Config:** `config.yaml` at repo root. Frozen dataclasses — never mutate at runtime.
- **4 GiB RAM cap** — respect in all runtime and benchmark paths.

## Conventions

- **Branch:** `feat/#num_of_issue-description-kebab-case`
- **Commit:** `<type>(#num_of_issue): short description`
- Small, descriptive commits. Don't rewrite history.

## History

The unified pipeline was merged from three independent prototypes (Milestone 1-2 work). The prototypes are archived in `slop/` — do not modify them.

- `slop/valenia/` — Valenia's pipeline (gallery, live runtime, tracking, metrics, tests, benchmarks)
- `slop/shalaiev/` — Shalaiev's pipeline (insightface SCRFD, UltraFace, FAISS+SQLite, display)
- `slop/avdieienko/` — Avdieienko's pipeline (YAML config, clean architecture, DI wiring)

For detailed attribution of what came from where: `docs/ATTRIBUTIONS.md`
For previous milestone reports: `slop/*/docs/`
