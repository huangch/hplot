# H-Plot reproducibility container
#
# Pure-CPU plotting + stats core — no GPU/CUDA stack needed, so this uses a
# slim base + pip rather than the conda/CUDA image the wsinsight/sptxinsight
# siblings use. pygam ships a pure-Python wheel, so no compiler is required.
#
# Build:  ./docker-build-push.sh              # or: docker build -t hplot:latest .
# Run (interactive shell; uid auto-matched to the mounted /workspace owner):
#   docker run --rm -it -v "$PWD":/workspace hplot:latest
# Run (a CLI command, as the remapped user):
#   docker run --rm -v "$PWD":/workspace hplot:latest \
#       hplot test -i /workspace/data.csv --target immune_fraction \
#                  --group hpv_status --permutations 999
# Force a specific uid/gid instead of the mount owner:
#   docker run --rm -e HOST_UID=1000 -e HOST_GID=1000 \
#       -v "$PWD":/workspace hplot:latest bash

FROM python:3.11-slim

LABEL org.opencontainers.image.title="hplot" \
      org.opencontainers.image.description="H-Plot: graph-geodesic spatial profiling at tissue boundaries" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# bash (for the entrypoint + default shell) and util-linux (setpriv, used by the
# run-time uid-remap entrypoint) are not in the slim base by default.
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash util-linux \
 && rm -rf /var/lib/apt/lists/*

# Core deps declared in pyproject.toml (matplotlib/pandas/scipy/numpy/pygam/
# anndata). Must be kept in sync with [project.dependencies] — the package
# itself is installed with --no-deps below, so anything missing here is missing
# in the image.
# NOTE: the version specs are QUOTED so the shell does not treat `>=` as a file
# redirect — the previous unquoted form silently dropped every pin. Installed in
# their own layer so the (rarely-changing) dependency cache survives source edits.
# constraints.txt is copied first so this layer is locked to the same set the
# conda env uses; unconstrained, pip could pick a pygam whose scipy range
# conflicts with the rest of the stack.
COPY constraints.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -c constraints.txt \
        "matplotlib>=3.0" \
        "pandas>=1.0" \
        "scipy>=1.6" \
        "numpy>=1.18" \
        "pygam>=0.9" \
        "anndata>=0.11,<0.13"

# MCP server (hplot-mcp). Shipped in the image so the MCP server works both in
# the conda env (conda-setup.sh hplot --mcp) and in Docker without an extra install.
# Kept in its own layer so core-only rebuilds don't re-fetch it.
RUN pip install --no-cache-dir -c constraints.txt "fastmcp>=3.4.5"

# Install the package itself from pyproject.toml (authoritative). The legacy
# setup.py is NOT copied — it is a stale duplicate whose install_requires is
# missing pygam.
COPY pyproject.toml README.md ./
COPY hplot/ ./hplot/
RUN pip install --no-cache-dir --no-deps .

# Build-time sanity: the CLI, the MCP server, and the import must work before
# we bake the image.
RUN hplot --help >/dev/null \
 && hplot-mcp --help >/dev/null \
 && python -c "import hplot; print('hplot', hplot.__version__, 'OK')"

# Non-root user. uid is 1000 (matching the siblings); it is remapped at RUN time
# by the entrypoint to the owner of the mounted /workspace (or $HOST_UID/$HOST_GID),
# so the baked id never has to match the host.
COPY docker-entrypoint.sh ./
RUN groupadd -g 1000 user \
 && useradd -m -u 1000 -g 1000 user \
 && install -m 0755 ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace
# NOTE: no `USER` here on purpose — the container starts as root so the
# entrypoint can remap `user` to the mount owner, then drops privileges via
# setpriv. `docker run --user ...` still works: the entrypoint detects a
# non-root start and execs the command unchanged.
SHELL ["/bin/bash", "-lc"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
