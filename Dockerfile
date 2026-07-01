# H-Plot reproducibility container
#
# Build:  docker build -t hplot .
# Run:    docker run --rm -v "$PWD":/data hplot \
#             test -i /data/data.csv --target immune_fraction \
#                  --group hpv_status --permutations 999

FROM python:3.11-slim

LABEL org.opencontainers.image.title="hplot" \
      org.opencontainers.image.description="H-Plot: graph-geodesic spatial profiling at tissue boundaries" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install build deps then package deps in one layer; no cache kept
COPY pyproject.toml setup.py* README.md ./
COPY hplot/ ./hplot/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        matplotlib>=3.0 \
        pandas>=1.0 \
        scipy>=1.6 \
        numpy>=1.18 \
        pygam>=0.9 \
 && pip install --no-cache-dir --no-deps -e .

# Non-root user for security
RUN useradd -m hplotuser
USER hplotuser

ENTRYPOINT ["hplot"]
CMD ["--help"]
