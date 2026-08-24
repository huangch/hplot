#!/usr/bin/env bash
# conda-setup.sh — create and populate the standalone hplot conda environment.
#
# Usage:  sh ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-m|--mcp]
#
#   -n | --name  ENV_NAME   Conda environment to use (default: current active env).
#   -r | --reset            Deactivate, remove, recreate, and activate the env.
#                           Without this flag the script skips env creation and
#                           only (re-)installs packages into the existing env.
#   -m | --mcp              Also install the `mcp` extra (fastmcp) which
#                           provides the `hplot-mcp` server. NOT installed by
#                           default (matching the wsinsight/sptxinsight
#                           convention) to keep the env lean.
#
# hplot is the H-Plot stats/plotting core. It is pure CPU (no GPU/CUDA stack),
# so this script is intentionally lean.

set -e   # abort on first error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
ENV_NAME="${CONDA_DEFAULT_ENV:-}"   # default = current active env
DO_RESET=0
DO_MCP=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--name)
            if [ -z "${2:-}" ]; then
                echo "Error: -n/--name requires an environment name." >&2
                exit 1
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        -r|--reset)
            DO_RESET=1
            shift
            ;;
        -m|--mcp)
            DO_MCP=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: sh ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-m|--mcp]" >&2
            exit 1
            ;;
    esac
done

if [ -z "$ENV_NAME" ]; then
    echo "Error: no conda environment specified and no environment is currently active." >&2
    echo "       Use -n ENV_NAME to specify one." >&2
    exit 1
fi

echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, mcp=${DO_MCP})"

# ── (Re-)create environment ───────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "${CONDA_BASE}" ]; then
    for _base in /opt/conda /opt/anaconda3; do
        if [ -f "${_base}/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="${_base}"
            break
        fi
    done
fi
if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "Error: cannot locate conda.sh. Activate conda first or set CONDA_BASE." >&2
    exit 1
fi
. "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ "$DO_RESET" -eq 1 ]; then
    conda deactivate
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    # Pure CPU stack — no CUDA/GPU packages.
    conda create -n "${ENV_NAME}" python=3.11 -c conda-forge -y
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# ── Pip cache fix (NAS inode quota) ──────────────────────────────────────────
# Redirect pip's wheel cache to /tmp to bypass NAS inode quotas. Exported before
# any purge: `pip cache purge` obeys this variable, so purging first wiped the
# user's global ~/.cache/pip.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-hplot}"

# ── Install hplot + core deps from pyproject.toml ─────────────────────────────
# Core = matplotlib/pandas/scipy/numpy/pygam/anndata. Only the mcp extra is
# opt-in (-m/--mcp), so the default env stays lean.
# constraints.txt is the lockfile; pyproject keeps loose bounds on purpose, so
# without -c pip is free to pick e.g. a pygam whose scipy range conflicts.
CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"
if [ "${DO_MCP}" -eq 1 ]; then
    pip install -c "${CONSTRAINTS}" -e "${SCRIPT_DIR}[mcp]"
else
    pip install -c "${CONSTRAINTS}" -e "${SCRIPT_DIR}"
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
# Hard checks are fatal: a half-installed env must not look like a success.
# The test suite is reported but does not fail the setup.
echo "---- smoke test ----"
SMOKE_FAIL=0
smoke() {                       # smoke <label> <command...>
    label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  PASS  %s\n' "$label"
    else
        printf '  FAIL  %s\n' "$label"
        SMOKE_FAIL=$((SMOKE_FAIL + 1))
    fi
}

python -c 'import importlib.metadata as m; print("  numpy", m.version("numpy"), "| scipy", m.version("scipy"), "| pygam", m.version("pygam"))' || true

smoke "hplot on PATH"        command -v hplot
smoke "hplot --help"         hplot --help
smoke "import hplot"         python -c 'import hplot'
smoke "import pygam"         python -c 'import pygam'
smoke "AnnData API usable"   python -c 'import hplot.pp, anndata'
# Only matters when this env is shared with wsinsight; hplot alone tolerates 2.x.
smoke "numpy < 2"            python -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)'
if [ "${DO_MCP}" -eq 1 ]; then
    smoke "hplot-mcp on PATH"    command -v hplot-mcp
    smoke "hplot-mcp --help"     hplot-mcp --help
fi

# NOTE: this repo uses test/, not tests/.
if [ -d "${SCRIPT_DIR}/test" ]; then
    if python -c "import pytest" >/dev/null 2>&1; then
        python -m pytest "${SCRIPT_DIR}/test" -q \
            && echo "  PASS  test suite" \
            || echo "  WARN  test suite did not pass (non-fatal)"
    else
        echo "  SKIP  test suite (pytest not installed; pip install -e '.[dev]')"
    fi
fi

if [ "${SMOKE_FAIL}" -ne 0 ]; then
    echo "smoke test: ${SMOKE_FAIL} check(s) FAILED" >&2
    exit 1
fi
echo "smoke test: all checks passed"
