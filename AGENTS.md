# hplot — Agent Guide

H-Plot: Kaplan-Meier-style curves of tissue composition vs **signed** distance from a tissue boundary (e.g. tumor–stroma interface). Layer 0 = boundary, >0 inside, <0 outside; per-layer CI across cases. Python >=3.7, Apache-2.0, v0.1.0.

## What this package is

The **stats/plotting core** of the ecosystem. It is a library + thin CLI, not a data pipeline:

- `wsinsight` (sibling) calls into it for its `hplot` / `hplot-finalize` CLI commands.
- `sptxinsight` (sibling) **vendors** the engine under `sptxinsight.insightlib` — changes here do NOT propagate automatically; the vendored copy must be re-synced by hand.

## Layout

- `cli.py` — `hplot` entry point (`hplot.cli:main`).
- `core.py` — `HPlot` fit (Stage 0: per-layer mean ± CI).
- `stats.py` — inference: `gradient_cluster_mass_screen` (Stage 1: Mann-Whitney U per layer, contiguous significant runs as statistic, slide-level sign-flip permutation null, FDR), `deviation_tensor`, `directional_cluster_bands`.
- `pathways.py` — H-Pathway: rank-based `ucell_scores` + `pathway_layer_profile`. **Deliberately no self-contained per-cell UCell-average test** — that's false-positive-prone on targeted panels. Scoring is separated from inference on purpose; don't "simplify" it back.
- `plotting.py` — `plot_hloci_bands/strip/fdr/dotplot`, `plot_hpathway_dotplot`, etc.
- `tl.py` / `pp.py` / `pl.py` — scanpy-style layers; `_anndata.py` — AnnData I/O.
- `mcp/` — `hplot-mcp` FastMCP server (optional `mcp` extra).

## The three stages (don't conflate)

1. **Stage 0** — per-layer mean ± CI (`HPlot.fit()`).
2. **Stage 1** — cluster-mass permutation test (handles spatial autocorrelation).
3. **Stage 2** — GAM effect size + confounder adjustment (pygam, penalised B-spline, GCV smoothing).

H-Loci summary: `deviation_tensor` → cross-slide signed z → cluster-mass bands → sign-flip null → FDR; its `gene_bands` table feeds H-Pathway ORA (`hpathway_layer_ora` = competitive counting of flagged genes per pathway).

## How it differs from squidpy `var_by_distance`

squidpy: unsigned distance from anchor points + polynomial fit, descriptive only. hplot: **signed** border layers + **cohort-level** inference (CI, permutation, GAM). If asked "why not just use squidpy", that's the answer.

## MCP server (`hplot-mcp`)

- Entry point `hplot.mcp.__main__:main`; extra `mcp = ["fastmcp>=2.0"]`. stdio by default; `--http HOST:PORT` (suggested port **8767**, after wsinsight 8765 / sptxinsight 8766). `--max-concurrent N` (default 1 — pure CPU).
- **Unlike wsinsight (bundled JSON) and sptxinsight (live `describe`), hplot has no `describe` command** — the tool surface is a **hand-written** command table in `hplot/mcp/schema.py` (one entry per sub-command). **Keep it in sync with `hplot/cli.py`** whenever the CLI changes.
- Tools: one per sub-command, faithful per-parameter mirrors of `hplot <sub> --help`. Long-running (`test`, `screen`, `loci` — permutation-heavy) return a `job_id`; poll `job_status` / `job_logs` / `cancel_job` / `list_jobs`. Short (`plot`, `gam`) run synchronously (600 s timeout) and return `{status, returncode, argv, duration_s, log_tail}`.
- Resource `hplot://schema` (the command table) + prompt `hplot_workflow`.
- Adapter (`hplot/mcp/adapters.py`) translates snake_case args → kebab-case `--flags`; bool flags only when truthy; `nargs` args repeated; no positional args.

## Tests

- `python -m pytest test/` (note: `test/`, not `tests/`).
- No CI workflow in this repo (no `.github/`); it's validated from the sibling repos' pipelines.

## Environment

- Standalone env: `sh ./conda-setup.sh -n hplot [-r|--reset] [-e|--extras] [-m|--mcp]` — creates a py3.11 env with the core deps (matplotlib/pandas/scipy/numpy/pygam). No GPU/CUDA stack needed (pure CPU plotting + stats). The `-m`/`--mcp` flag adds `fastmcp` (the `hplot-mcp` server); **not installed by default** (matching the wsinsight/sptxinsight convention).
- Docker: `./docker-build-push.sh` builds `hplot:latest` and pushes `huangchtw/hplot:latest`. The image ships a `user` (uid 1000) and an entrypoint that remaps it to the mount owner at run time (same pattern as wsinsight). `fastmcp` is baked in, so `hplot-mcp` works in the image without an extra install.

## Conventions

- Deps are minimal on purpose: matplotlib/pandas/scipy/numpy/pygam. `anndata` and `squidpy` are **optional extras**, not core — guard imports.
- No lint config in `pyproject.toml` (no ruff/pytest sections); keep style consistent with existing modules.

## Sibling repos (same ecosystem)

- `wsinsight` — WSI pipeline that calls this package (H-Plot CLI).
- `sptxinsight` — spatial-transcriptomics sibling with a vendored copy of this engine.
- `clawsight` / `clawpyter` — client-side agent plugins for the ecosystem's MCP servers / Jupyter.
