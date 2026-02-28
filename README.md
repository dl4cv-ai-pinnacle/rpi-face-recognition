# rpi-face-recognition

The active implementation now lives under `slop/valenia/`.

## Start Here

1. Read the setup guide:

```bash
sed -n '1,220p' slop/valenia/README.md
```

2. Bootstrap the Pi:

```bash
./slop/valenia/scripts/bootstrap_pi.sh
```

3. Run quality checks from the repo root:

```bash
uv run --project slop/valenia --group dev pyright -p slop/valenia --pythonpath /usr/bin/python3
uv run --project slop/valenia --group dev pre-commit run \
  --config slop/valenia/.pre-commit-config.yaml \
  --all-files
```

## Repo Layout

- Active setup: `slop/valenia/`
- Milestone brief: `MILESTONE_1.md`
- Repo agent rules: `AGENTS.md`
