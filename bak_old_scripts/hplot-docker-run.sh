#!/bin/sh

IMAGE_ID=hplot:latest
docker pull ${IMAGE_ID}

# The container's uid/gid is set at run time by the image entrypoint: by default
# it becomes the owner of the mounted /workspace (so you can always write to your
# data). Export HOST_UID / HOST_GID before running to force a specific id; the
# ``-e HOST_UID -e HOST_GID`` on the docker run lines forward them only when set.
#
# hplot is a pure-CPU plotting/stats package: no --gpus, no --shm-size, no HF
# model cache. TMPDIR defaults to /tmp (standard Linux default); if the
# container's /tmp has limited space, use --tmpdir /workspace/.tmp to redirect
# temp files (and large figure output) to the mounted /workspace filesystem.

# Parse arguments: --tmpdir <dir> is optional and may appear anywhere before the command.
DATA_DIR=""
TMPDIR_FLAG="/tmp"

while [ $# -gt 0 ]; do
    case "$1" in
        --tmpdir)
            TMPDIR_FLAG="$2"
            shift 2
            ;;
        --tmpdir=*)
            TMPDIR_FLAG="${1#--tmpdir=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            # First non-option arg is the data dir; remaining are the command.
            if [ -z "${DATA_DIR}" ]; then
                DATA_DIR="$1"
                shift
            else
                break
            fi
            ;;
    esac
done

if [ -z "${DATA_DIR}" ]; then
    echo "Usage: hplot-docker-run.sh [--tmpdir <DIR>] /path/to/data [COMMAND ...]"
    echo ""
    echo "Options:"
    echo "  --tmpdir <DIR>  Override temp directory (default: /tmp)"
    echo ""
    echo "Examples:"
    echo "  hplot-docker-run.sh /data                              # interactive shell"
    echo "  hplot-docker-run.sh --tmpdir /workspace/.tmp /data     # use /workspace for temp files"
    echo "  hplot-docker-run.sh /data hplot test -i /workspace/data.csv --target immune_fraction --group hpv_status --permutations 999"
    exit 1
fi

if [ $# -gt 0 ]; then
    # Direct command mode: run the provided command and exit
    echo docker run --rm -it --init -e HOST_UID -e HOST_GID -e TMPDIR="${TMPDIR_FLAG}" -v "${DATA_DIR}":/workspace ${IMAGE_ID} bash -lc "$*"
    docker run --rm -it --init -e HOST_UID -e HOST_GID -e TMPDIR="${TMPDIR_FLAG}" -v "${DATA_DIR}":/workspace ${IMAGE_ID} bash -lc "$*"
else
    # Interactive mode: drop into a shell
    echo docker run --rm -it --init -e HOST_UID -e HOST_GID -e TMPDIR="${TMPDIR_FLAG}" -v "${DATA_DIR}":/workspace ${IMAGE_ID}
    docker run --rm -it --init -e HOST_UID -e HOST_GID -e TMPDIR="${TMPDIR_FLAG}" -v "${DATA_DIR}":/workspace ${IMAGE_ID}
fi
