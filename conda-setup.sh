#!/usr/bin/env bash
# conda-setup.sh — create and populate the standalone hplot conda environment.
#
# <<<USAGE_START>>>
# Usage:  sh ./conda-setup.sh ENV_NAME [-r|--reset] [-m|--mcp] [-d|--dev] [-h|--help]
#
#   ENV_NAME                (positional, REQUIRED) Conda environment to use/create.
#                           There is NO fallback to the currently-activated conda env:
#                           the name is mandatory so `-r` can never accidentally
#                           destroy a different active environment.
#   -r | --reset            Deactivate, remove, recreate, and activate ENV_NAME.
#                           Without this flag the script skips env creation and
#                           only (re-)installs packages into the existing env.
#   -m | --mcp              Also install the `mcp` extra (fastmcp) which
#                           provides the `hplot-mcp` server. NOT installed by
#                           default (matching the wsinsight/sptxinsight
#                           convention) to keep the env lean.
#   -d | --dev              Also install the [dev] extra (pytest, pytest-cov,
#                           pre_commit) so the post-install smoke test can run the
#                           real test suite. Without -d the suite is SKIPped if
#                           pytest is missing; with -d it FAILS (you asked for it).
#                           The package itself is always installed editable (-e).
#   -h | --help             Print this help message and exit.
# <<<USAGE_END>>>
#
# hplot is the H-Plot stats/plotting core. It is pure CPU (no GPU/CUDA stack),
# so this script is intentionally lean.

set -e   # abort on first error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
# ENV_NAME is the FIRST POSITIONAL argument and is REQUIRED. We deliberately do
# NOT fall back to $CONDA_DEFAULT_ENV: with `-r` in play, a hidden dependency on
# whatever env happens to be active is a footgun (it would silently destroy an
# unrelated env). Make the caller name the env explicitly, every time.
DO_RESET=0
DO_MCP=0
DO_DEV=0

print_usage() {
    awk '
        /<<<USAGE_START>>>/ {capture=1; next}
        /<<<USAGE_END>>>/   {capture=0}
        capture            {sub(/^# ?/, ""); print}
    ' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -r|--reset)
            DO_RESET=1
            shift
            ;;
        -m|--mcp)
            DO_MCP=1
            shift
            ;;
        -d|--dev)
            DO_DEV=1
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Run '${0##*/} --help' for usage." >&2
            exit 1
            ;;
        *)
            # First non-flag token is ENV_NAME. Reject a second positional
            # argument (we only have one positional slot).
            if [ -n "${ENV_NAME:-}" ]; then
                echo "Error: only one positional argument (ENV_NAME) is accepted; got '$ENV_NAME' and '$1'." >&2
                echo "Run '${0##*/} --help' for usage." >&2
                exit 1
            fi
            ENV_NAME="$1"
            shift
            ;;
    esac
done

if [ -z "${ENV_NAME:-}" ]; then
    echo "Error: ENV_NAME is required." >&2
    echo "       Got: $0 $*" >&2
    echo "       Run '${0##*/} --help' for usage." >&2
    exit 1
fi

echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, mcp=${DO_MCP}, dev=${DO_DEV})"

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
# Core = matplotlib/pandas/scipy/numpy/pygam/anndata. Only the mcp and dev
# extras are opt-in (-m/--mcp, -d/--dev), so the default env stays lean.
# constraints.txt is the lockfile; pyproject keeps loose bounds on purpose, so
# without -c pip is free to pick e.g. a pygam whose scipy range conflicts.
# With -d/--dev, also install the [dev] extra (pytest, pytest-cov, pre_commit)
# so the smoke test can run the suite; without -d, the suite is SKIPped if
# pytest is missing and only WARN-ed if it fails.
CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"
EXTRAS=""
[ "${DO_MCP}" -eq 1 ] && EXTRAS="${EXTRAS},mcp"
[ "${DO_DEV}" -eq 1  ] && EXTRAS="${EXTRAS},dev"
if [ -n "${EXTRAS}" ]; then
    pip install -c "${CONSTRAINTS}" -e "${SCRIPT_DIR}[${EXTRAS#,}]"
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
    elif [ "${DO_DEV}" -eq 1 ]; then
        # User asked for the [dev] extra: pytest should be present. FAIL loudly
        # instead of silently SKIPping, or the install is misconfigured.
        echo "  FAIL  test suite: pytest missing but -d/--dev was requested" >&2
        smoke "pytest importable (dev)" python -c "import pytest"
    else
        echo "  SKIP  test suite (pytest not installed; rerun with -d/--dev)"
    fi
fi

if [ "${SMOKE_FAIL}" -ne 0 ]; then
    echo "smoke test: ${SMOKE_FAIL} check(s) FAILED" >&2
    exit 1
fi
echo "smoke test: all checks passed"
