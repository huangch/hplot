# bak_old_scripts/

Legacy hplot runner scripts that have been **superseded** by [`hplot.sh`](../hplot.sh) at the repo root.

## What moved here

| File | Date moved | Reason |
|---|---|---|
| `hplot-docker-run.sh` | 2026-09-05 | Folded into `hplot.sh --runner docker`. The new wrapper supersedes this script and fixes its `IMAGE_ID=hplot:latest` (now `huangchtw/hplot:latest`). |

## Why kept

- Source-of-truth for the historical docker invocation pattern (mount structure, env flags, TMPDIR override).
- The legacy script's `IMAGE_ID=hplot:latest` references an **un-published** local-only image; the new wrapper uses **`huangchtw/hplot:latest`** (matching Docker Hub).
- Useful if you need to reproduce exactly what the old wrapper did; **not** the recommended entry point going forward.

## Migration

Replace
```
bash hplot-docker-run.sh /path/to/data --tmpdir /workspace/.tmp hplot screen -i raw.csv -o ranking.csv
```
with
```
export HPLOT_DATA_DIR=/path/to/data
./hplot.sh --runner docker --tmpdir /workspace/.tmp screen -i raw.csv -o ranking.csv
```

Anything you used to pass through the legacy script after the data dir is now passed through `./hplot.sh` after the first hplot subcommand name (the wrapper auto-detects the boundary).
