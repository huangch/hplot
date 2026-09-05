#!/usr/bin/env bash
# hplot.sh - unified environment-aware runner for the hplot CLI.
#
# Manages BOTH runners (native + docker) without depending on the legacy
# hplot-docker-run.sh script, which has moved to bak_old_scripts/.
#
# Subcommands:
#   run        --runner {native,docker} [--tmpdir DIR] [--no-pull]
#               [--dry-run] [HPLOT_ARGS ...]
#   status                                                  # current effective config
#   doctor     [--runner {native,docker}]                  # preflight
#   where                                                   # absolute path of this script
#   -h | --help
#   --version
#
# Why hplot's wrapper is simpler than the wsinsight/sptxinsight equivalents:
#   hplot is pure-CPU (no --gpu / --shm-size), has no HF model cache (no
#   persistent volume), and writes figures to a data dir, not a separate
#   results dir. Only --tmpdir and --no-pull are the docker-only env knobs.
#
# Why --runner (not -b / --backend):
#   hplot's CLI does not currently have a global --backend flag, so we could
#   have used -b. We keep --runner for cross-tool uniformity with the
#   sptxinsight wrapper; once you learn the wrapper grammar for one tool, it
#   transfers.
#
# Param-parsing rule:
#   Everything before the first hplot subcommand name (plot, test, gam, screen,
#   loci, schema) is consumed by THIS script (env control: --runner, --tmpdir,
#   --no-pull, --dry-run). From (and including) the first hplot subcommand
#   name, every token is passed through verbatim.
#
#   Boundary discovery (same rules as wsinsight.sh / sptxinsight.sh):
#     1. Absorb the script's own flags and subcommands (status / doctor /
#        where take priority over hplot subcommands of the same name).
#     2. Honor an explicit `--` delimiter.
#     3. The first position arg is checked against the list of known hplot
#        subcommand names; if it matches, passthrough begins THERE.
#     4. Backward-scan fallback: if the first position arg is not a known
#        subcommand, scan the remainder of argv for the first known one.
#     5. If no known hplot subcommand is found anywhere, die with the list.
#
#   Unknown script flags (-X that the wrapper does not recognize): in default
#   (lenient) mode, warn and treat as hplot's; in HPLOT_STRICT=1 mode, die.
#
# Exit code 0 on success, 1 on user error, 2 on infrastructure failure.

set -euo pipefail

PROG="$(basename "$0")"
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IMAGE_ID="${HPLOT_IMAGE:-huangchtw/hplot:latest}"
COMMANDS_CACHE="$HOME/.cache/hplot/commands.txt"
COMMANDS_CACHE_TTL_SECONDS="${HPLOT_COMMANDS_TTL_SECONDS:-86400}"

# `-d` / `--data-dir` is REQUIRED when --runner docker is in effect. It names
# the host dir bind-mounted to /workspace inside the container; mirrors
# wsinsight.sh and sptxinsight.sh. HPLOT_DATA_DIR is the matching env var
# (kept for non-interactive use; the flag takes precedence).
SCRIPT_DATA_DIR="${HPLOT_DATA_DIR:-}"

# LAST-RESORT builtin subcommand list - updated as hplot evolves.
# Used only when neither the cache nor a live `hplot schema --commands-only`
# works. The hand-written table in hplot/mcp/schema.py stays the source of
# truth; this is a static fallback for offline / broken-shell scenarios.
_HPLOT_BUILTIN_CMDS=(
    plot test gam screen loci
)

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
print_usage() {
    cat <<EOF
$PROG $VERSION - run hplot with one of two runners (native | docker)

Usage:
  $PROG run     --runner {native,docker} -d DIR | --data-dir DIR
                [--tmpdir DIR] [--no-pull] [--dry-run] [HPLOT_ARGS ...]
  $PROG status
  $PROG doctor  [--runner {native,docker}]
  $PROG where
  $PROG -h | --help
  $PROG --version

Runners (selected via --runner):
  native    Invoke the hplot CLI on the host inside the activated env.
  docker    Run hplot inside the $IMAGE_ID container (no --gpu / no model-cache).
            Requires -d DIR (the host dir bind-mounted to /workspace inside the
            container). HPLOT_DATA_DIR provides the same value non-interactively.

Environment overrides:
  HPLOT_RUNNER                 Default runner when --runner is not given (native | docker)
  HPLOT_IMAGE                  Override the docker image tag
  HPLOT_COMMANDS_TTL_SECONDS   TTL for cached subcommand list (default 86400)
  HPLOT_DATA_DIR               Default -d value when the flag is not given (used by docker)
  HPLOT_STRICT=1               Die on unknown -X flags instead of warning + passthrough

Decision rule for argv parsing:
  Everything between the script's own flags and the first hplot subcommand
  name (plot/test/gam/screen/loci) is consumed by this script. From (and
  including) the first hplot subcommand name, every remaining argument is
  passed verbatim to hplot. Use -- to force passthrough explicitly.

Examples:
  $PROG plot -i data.csv --target immune_fraction --group hpv_status -o out/    # native (default)
  $PROG --runner docker -d /workspace/project plot -i /workspace/data.csv -o /workspace/out/
  $PROG --runner docker -d /workspace/project --tmpdir /scratch --no-pull screen -i raw.csv -o ranking.csv
  $PROG --runner docker -d /workspace/project --dry-run screen -i raw.csv    # print resolved docker command, do not run
  $PROG doctor
  $PROG where
EOF
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_log()  { printf '[%s] %s\n' "$PROG" "$*" >&2; }
_warn() { _log "WARNING: $*"; }
_die()  { _log "ERROR: $*"; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# hplot subcommand discovery
# ---------------------------------------------------------------------------
_get_hplot_command_list() {
    if [[ -f "$COMMANDS_CACHE" ]]; then
        local cache_age
        cache_age=$(( $(date +%s) - $(stat -c %Y "$COMMANDS_CACHE" 2>/dev/null || echo 0) ))
        if [[ $cache_age -lt $COMMANDS_CACHE_TTL_SECONDS ]]; then
            cat "$COMMANDS_CACHE"
            return 0
        fi
    fi

    if command -v hplot >/dev/null 2>&1; then
        local out
        if out="$(hplot schema --commands-only 2>/dev/null)" \
           && [[ -n "$out" ]] \
           && command -v python3 >/dev/null 2>&1; then
            local extracted
            if extracted="$(printf '%s' "$out" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    cmds = d.get('commands') or []
    if isinstance(cmds, dict): cmds = list(cmds.keys())
    for c in cmds: print(c)
except Exception:
    sys.exit(1)
")" && [[ -n "$extracted" ]]; then
                mkdir -p "$(dirname "$COMMANDS_CACHE")"
                printf '%s\n' "$extracted" > "$COMMANDS_CACHE"
                printf '%s\n' "$extracted"
                return 0
            fi
        fi
    fi

    # LAST RESORT
    _warn "could not refresh subcommand cache; using builtin list (last known good)"
    printf '%s\n' "${_HPLOT_BUILTIN_CMDS[@]}"
}

_is_hplot_cmd() {
    local needle="$1" cmd
    for cmd in "${HPLOT_CMDS[@]}"; do
        [[ "$cmd" == "$needle" ]] && return 0
    done
    return 1
}

# Load subcommand list (once per invocation)
HPLOT_CMDS=()
while IFS= read -r c; do
    [[ -n "$c" ]] && HPLOT_CMDS+=("$c")
done < <(_get_hplot_command_list)
[[ ${#HPLOT_CMDS[@]} -gt 0 ]] || _die "could not determine hplot subcommand list (cache + fallback both empty)"

# ---------------------------------------------------------------------------
# Phase 1: parse argv
# ---------------------------------------------------------------------------
SCRIPT_RUNNER=""
SCRIPT_TMPDIR=""
SCRIPT_NO_PULL=0
SCRIPT_DRY_RUN=0
EXTRA_CMD=""           # status | doctor | where

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner)
            [[ -n "${2:-}" ]] || _die "--runner requires a value"
            SCRIPT_RUNNER="$2"; shift 2 ;;
        --runner=*)
            SCRIPT_RUNNER="${1#*=}"; shift ;;
        --tmpdir)
            [[ -n "${2:-}" ]] || _die "--tmpdir requires a value"
            SCRIPT_TMPDIR="$2"; shift 2 ;;
        --tmpdir=*)
            SCRIPT_TMPDIR="${1#*=}"; shift ;;
        -d|--data-dir)
            [[ -n "${2:-}" ]] || _die "-d / --data-dir requires a value"
            SCRIPT_DATA_DIR="$2"; shift 2 ;;
        -d=*|--data-dir=*)
            SCRIPT_DATA_DIR="${1#*=}"; shift ;;
        --no-pull)
            SCRIPT_NO_PULL=1; shift ;;
        --dry-run|--dryrun)
            SCRIPT_DRY_RUN=1; shift ;;
        -h|--help)
            print_usage; exit 0 ;;
        --version)
            echo "$PROG $VERSION"; exit 0 ;;
        --)
            shift
            break ;;
        status|doctor|where)
            EXTRA_CMD="$1"; shift
            break ;;
        -*)
            if [[ "${HPLOT_STRICT:-0}" == "1" ]]; then
                _die "unknown option: $1 (HPLOT_STRICT=1; known hplot subcommands: ${HPLOT_CMDS[*]})"
            else
                _warn "unrecognized script flag: $1 -- treating as hplot's (use HPLOT_STRICT=1 to enforce)"
                break
            fi
            ;;
        *)
            if _is_hplot_cmd "$1"; then
                break
            fi
            # Backward-scan fallback
            matched=0; found_idx=0; i=1
            for arg in "$@"; do
                if _is_hplot_cmd "$arg"; then
                    matched=1; found_idx=$i; break
                fi
                i=$((i + 1))
            done
            if [[ $matched -eq 1 ]]; then
                shift $((found_idx - 1))
                break
            fi
            _die "no hplot subcommand found in: $* (known: ${HPLOT_CMDS[*]})"
            ;;
    esac
done
HPLOT_ARGS=("$@")

# ---------------------------------------------------------------------------
# Phase 2: dispatch on EXTRA_CMD
# ---------------------------------------------------------------------------

if [[ -z "$SCRIPT_RUNNER" ]]; then
    SCRIPT_RUNNER="${HPLOT_RUNNER:-native}"
fi
case "$SCRIPT_RUNNER" in
    native|docker) ;;
    *) _die "unknown runner: '$SCRIPT_RUNNER' (use 'native' or 'docker')" ;;
esac

cmd_status() {
    cat <<EOF
PROG         : $PROG ($SCRIPT_DIR/$PROG)
VERSION      : $VERSION
Runner       : $SCRIPT_RUNNER$( [[ $SCRIPT_DRY_RUN -eq 1 ]] && echo " (dry-run)" )
Data dir(-d) : ${SCRIPT_DATA_DIR:-<unset - required only for --runner docker>}
Tmpdir       : ${SCRIPT_TMPDIR:-<unchanged>}
No-pull      : $SCRIPT_NO_PULL
Image        : $IMAGE_ID
hplot        : $(command -v hplot 2>/dev/null || echo "<not on PATH>")
docker       : $(command -v docker 2>/dev/null || echo "<not on PATH>")
Subcommands  : ${#HPLOT_CMDS[@]} known; cache: $COMMANDS_CACHE
Pass-through : ${#HPLOT_ARGS[@]} arg(s)${HPLOT_ARGS[*]:+: ${HPLOT_ARGS[*]}}
EOF
}

cmd_doctor() {
    local target="${DOC_RUNNER:-$SCRIPT_RUNNER}"
    local rc=0
    printf 'doctor (runner=%s)\n' "$target"
    case "$target" in
        native)
            local hplt
            if hplt="$(command -v hplot)"; then
                printf '  [OK]  hplot on PATH: %s\n' "$hplt"
                if "$hplt" --version 2>/dev/null | grep -q "Options:\|Usage:"; then
                    printf '  [OK]  hplot --help runs\n'
                else
                    printf '  [WARN] hplot --help output unexpected\n'
                fi
            else
                printf '  [FAIL] hplot not on PATH (activate the hplot conda env, or use --runner docker)\n'
                rc=2
            fi
            ;;
        docker)
            if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
                printf '  [OK]  docker daemon reachable\n'
                if docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then
                    printf '  [OK]  image present locally: %s\n' "$IMAGE_ID"
                else
                    printf '  [INFO] image not local; will pull on first run: %s\n' "$IMAGE_ID"
                fi
            else
                printf '  [FAIL] docker not reachable\n'
                rc=2
            fi
            ;;
    esac
    return $rc
}

cmd_where() {
    echo "$SCRIPT_DIR/$PROG"
}

# Build the resolved docker run args (printed by --dry-run). Takes the host
# data dir as $1 so this function doesn't depend on main()'s scope.
build_docker_command() {
    local data_dir="$1"
    local parts=(docker run --rm -it --init)
    [[ -n "${HOST_UID:-}"   ]] && parts+=(-e HOST_UID)
    [[ -n "${HOST_GID:-}"   ]] && parts+=(-e HOST_GID)
    [[ -n "$SCRIPT_TMPDIR"  ]] && parts+=(-e TMPDIR="$SCRIPT_TMPDIR")
    parts+=(-v "$data_dir":/workspace)
    parts+=("$IMAGE_ID")
    if [[ ${#HPLOT_ARGS[@]} -gt 0 ]]; then
        parts+=(hplot "${HPLOT_ARGS[@]}")
    fi
    # Shell-quote each token (printf %q). Embedded-space tokens stay separate
    # args, which is what bash will see when re-parsing.
    local i
    for i in "${!parts[@]}"; do
        printf '%q' "${parts[$i]}"
        if [[ $i -lt $((${#parts[@]} - 1)) ]]; then printf ' '; fi
    done
}

case "$EXTRA_CMD" in
    where)   cmd_where; exit 0 ;;
    status)  cmd_status; exit 0 ;;
    doctor)  cmd_doctor; exit $? ;;
    "")      : ;;
    *)       _die "unknown subcommand: $EXTRA_CMD" ;;
esac

# ---------------------------------------------------------------------------
# Phase 3: execute
# ---------------------------------------------------------------------------

main() {
    if [[ ${#HPLOT_ARGS[@]} -eq 0 && -z "$EXTRA_CMD" ]]; then
        print_usage
        exit 0
    fi
    if [[ ${#HPLOT_ARGS[@]} -eq 0 ]]; then
        if [[ "$SCRIPT_RUNNER" == "native" ]]; then
            _die "no hplot subcommand supplied. Try: $PROG plot -i data.csv --target col --group col -o out/"
        fi
        # docker with no args -> interactive shell. Allowed by convention.
    fi

    case "$SCRIPT_RUNNER" in
        native)
            if [[ $SCRIPT_DRY_RUN -eq 1 ]]; then
                local hplt_bin_dry
                if [[ -n "${HPLOT_BIN:-}" ]]; then
                    hplt_bin_dry="$HPLOT_BIN"
                elif command -v hplot >/dev/null 2>&1; then
                    hplt_bin_dry="$(command -v hplot)"
                else
                    hplt_bin_dry="/opt/anaconda3/envs/hplot/bin/hplot"
                fi
                printf '+ %q %q\n' "$hplt_bin_dry" "${HPLOT_ARGS[*]}"
                exit 0
            fi
            local hplt_bin
            if [[ -n "${HPLOT_BIN:-}" ]]; then
                hplt_bin="$HPLOT_BIN"
            elif command -v hplot >/dev/null 2>&1; then
                hplt_bin="$(command -v hplot)"
            else
                hplt_bin="/opt/anaconda3/envs/hplot/bin/hplot"
            fi
            if [[ ! -x "$hplt_bin" ]]; then
                _die "hplot interpreter not found at '$hplt_bin'. Activate an env with hplot installed or set HPLOT_BIN."
            fi
            exec "$hplt_bin" "${HPLOT_ARGS[@]}"
            ;;
        docker)
            # Determine data dir: required for docker. The flag (-d / --data-dir)
            # takes precedence; HPLOT_DATA_DIR is the env-var form.
            local DATA_DIR="$SCRIPT_DATA_DIR"
            if [[ -z "$DATA_DIR" ]]; then
                _die "docker runner needs a data dir. Pass -d DIR (or --data-dir DIR), or set HPLOT_DATA_DIR=/path. The host dir is bind-mounted to /workspace inside the container; HPLOT_ARGS sees /workspace as cwd."
            fi
            if [[ ! -d "$DATA_DIR" ]]; then
                _die "--data-dir '$DATA_DIR' does not exist (or is not a directory). Pass -d DIR pointing at an existing path."
            fi
            if [[ $SCRIPT_DRY_RUN -eq 1 ]]; then
                echo "+ $(build_docker_command "$DATA_DIR")"
                exit 0
            fi
            if [[ $SCRIPT_NO_PULL -eq 0 ]]; then
                docker pull "$IMAGE_ID" >/dev/null 2>&1 \
                    || _warn "docker pull failed; using local image (if any)"
            fi
            local docker_args=( run --rm -it --init )
            [[ -n "${HOST_UID:-}"   ]] && docker_args+=( -e HOST_UID )
            [[ -n "${HOST_GID:-}"   ]] && docker_args+=( -e HOST_GID )
            [[ -n "$SCRIPT_TMPDIR"  ]] && docker_args+=( -e TMPDIR="$SCRIPT_TMPDIR" )
            docker_args+=( -v "$DATA_DIR":/workspace )
            docker_args+=( "$IMAGE_ID" )
            if [[ ${#HPLOT_ARGS[@]} -gt 0 ]]; then
                docker_args+=( hplot "${HPLOT_ARGS[@]}" )
            fi
            exec docker "${docker_args[@]}"
            ;;
    esac
}

main "$@"
