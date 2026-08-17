#!/usr/bin/env bash
# conda-setup.sh — create and populate the standalone hplot conda environment.
#
# Usage:  sh ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-e|--extras] [-m|--mcp]
#
#   -n | --name  ENV_NAME   Conda environment to use (default: current active env).
#   -r | --reset            Deactivate, remove, recreate, and activate the env.
#                           Without this flag the script skips env creation and
#                           only (re-)installs packages into the existing env.
#   -e | --extras           Also install the optional `anndata` + `squidpy`
#                           extras (AnnData API + squidpy-based examples).
#                           NOT installed by default — the core CLI is CSV-based
#                           and only needs matplotlib/pandas/scipy/numpy/pygam.
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
DO_EXTRAS=0
DO_MCP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            if [[ -z "${2:-}" ]]; then
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
        -e|--extras)
            DO_EXTRAS=1
            shift
            ;;
        -m|--mcp)
            DO_MCP=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: sh ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-e|--extras] [-m|--mcp]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$ENV_NAME" ]]; then
    echo "Error: no conda environment specified and no environment is currently active." >&2
    echo "       Use -n ENV_NAME to specify one." >&2
    exit 1
fi

echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, extras=${DO_EXTRAS}, mcp=${DO_MCP})"

# ── (Re-)create environment ───────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "${CONDA_BASE}" ]]; then
    for _base in /opt/conda /opt/anaconda3; do
        if [[ -f "${_base}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${_base}"
            break
        fi
    done
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Error: cannot locate conda.sh. Activate conda first or set CONDA_BASE." >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ "$DO_RESET" -eq 1 ]]; then
    conda deactivate
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    # Pure CPU stack — no CUDA/GPU packages.
    conda create -n "${ENV_NAME}" python=3.11 -c conda-forge -y
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# ── Pip cache fix (NAS inode quota) ──────────────────────────────────────────
pip cache purge || true
# Redirect pip's wheel cache to /tmp to bypass NAS inode quotas.
export PIP_CACHE_DIR=/tmp/pip-cache-hplot

# ── Install hplot + core deps from pyproject.toml ─────────────────────────────
# Core = matplotlib/pandas/scipy/numpy/pygam. Optional extras (anndata, squidpy,
# mcp) are opt-in via -e/--extras and -m/--mcp so the default env stays minimal.
EXTRAS=""
if [[ "${DO_EXTRAS}" -eq 1 ]]; then
    EXTRAS="anndata,squidpy"
fi
if [[ "${DO_MCP}" -eq 1 ]]; then
    if [[ -n "${EXTRAS}" ]]; then
        EXTRAS="${EXTRAS},mcp"
    else
        EXTRAS="mcp"
    fi
fi
if [[ -n "${EXTRAS}" ]]; then
    pip install -e "${SCRIPT_DIR}[${EXTRAS}]"
else
    pip install -e "${SCRIPT_DIR}"
fi

# ── Safety checks ─────────────────────────────────────────────────────────────
python -c "
import numpy, pandas, scipy, matplotlib, pygam
print(f'numpy {numpy.__version__} | pandas {pandas.__version__} | scipy {scipy.__version__} OK')
import hplot
print('hplot import OK:', hplot.__name__)
"

# ── Smoke test ────────────────────────────────────────────────────────────────
hplot --help

if [[ "${DO_MCP}" -eq 1 ]]; then
    hplot-mcp --help
fi
