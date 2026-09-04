# hplot — Agent Guide

H-Plot: Kaplan-Meier-style curves of tissue composition vs **signed** distance from a tissue boundary (e.g. tumor–stroma interface). Layer 0 = boundary, >0 inside, <0 outside; per-layer CI across cases. Python >=3.11, Apache-2.0, v0.1.0.

## What this package is

The **user-facing analysis layer** of the ecosystem. It is a library + thin CLI, not
a data pipeline, and it sits *downstream* of the pipelines:

```
wsinsight / sptxinsight  ->  CSV / h5ad / GeoJSON on disk
                                     |
                                     v
                          hplot  (Jupyter environment)
                                     |
                                     v
                              clawpyter drives it
```

- Installed in the **JupyterLab environment**, where `clawpyter` drives it to analyse
  outputs the pipelines have already written.
- **The pipelines do NOT depend on this package.** `wsinsight` and `sptxinsight` each
  ship their own `insightlib/` and neither imports `hplot`. That is deliberate — do not
  "de-duplicate" by making a pipeline import this package; it would invert the layering.
- Their `hplot` / `hplot-finalize` subcommands are a **name collision only**: those run
  the pipeline's own `insightlib.insight_helpers.compute_hplot`, which is a separate
  implementation from the `HPlot` class here.

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

- Standalone env: `sh ./conda-setup.sh hplot [-r|--reset] [-m|--mcp] [-d|--dev]` — creates a py3.11 env with the core deps (matplotlib/pandas/scipy/numpy/pygam/anndata). No GPU/CUDA stack needed (pure CPU plotting + stats). The `-m`/`--mcp` flag adds `fastmcp` (the `hplot-mcp` server); **not installed by default** (matching the wsinsight/sptxinsight convention). Add `-d`/`--dev` to also install pytest/pytest-cov/ruff/pre_commit for running the test suite; add `-r`/`--reset` to nuke and recreate the env. Run `./conda-setup.sh --help` for the full CLI.
- Docker: `./docker-build-push.sh` builds `hplot:latest` and pushes `huangchtw/hplot:latest`. The image ships a `user` (uid 1000) and an entrypoint that remaps it to the mount owner at run time (same pattern as wsinsight). `fastmcp` is baked in, so `hplot-mcp` works in the image without an extra install.

## Conventions

- Deps are minimal on purpose: matplotlib/pandas/scipy/numpy/pygam plus `anndata` (core, because `__init__` exports the `pp`/`tl`/`pl` API unconditionally). `anndata` is still imported **lazily inside functions** — `test/test_anndata_api.py` asserts no module-level import, so `import hplot.core` stays cheap. `squidpy` is not imported anywhere in the package; it is a convenience extra only.
- No lint config in `pyproject.toml` (no ruff/pytest sections); keep style consistent with existing modules.

## Sibling repos (same ecosystem)

- `wsinsight` — WSI pipeline. Produces the outputs analysed here; does **not** import
  this package.
- `sptxinsight` — spatial-transcriptomics sibling. Same relationship; its `insightlib/`
  was copied from wsinsight, not from this package.
- `clawsight` / `clawpyter` — client-side agent plugins for the ecosystem's MCP servers / Jupyter.
