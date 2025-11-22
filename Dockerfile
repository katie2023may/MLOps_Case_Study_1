# syntax=docker/dockerfile:experimental
########################################
# Multi-stage, multi-platform Dockerfile
# - builder: installs Python deps into /install
# - runtime: small final image, copies only runtime files
########################################

# Note: we avoid using --platform=$BUILDPLATFORM in the FROM lines to remain
# compatible with a wider range of builders. Use `docker buildx` with
# --platform when you need multi-arch manifests; buildx will handle emulation.

########################################
# Builder: install Python deps (kept out of final image)
########################################
FROM python:3.11-slim AS builder

WORKDIR /build
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools (only in builder)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements_docker.txt /build/requirements_docker.txt

RUN python -m pip install --upgrade pip setuptools wheel
# Install into an isolated prefix so we can copy only runtime files later
RUN python -m pip install --no-cache-dir --prefix=/install -r /build/requirements_docker.txt
########################################
# Runtime: minimal image
########################################
FROM python:3.11-slim AS runtime

WORKDIR /opt/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0

# Install runtime-only OS packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       prometheus-node-exporter \
       tini \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application files (use .dockerignore to exclude large files like model)
# Copy explicitly to keep layers small
COPY model_100.pth /opt/app/
COPY example1.png /opt/app/
COPY example2.png /opt/app/
COPY example3.png /opt/app/
# If you have additional small helper scripts, copy them explicitly, e.g.
# COPY server_connect.sh /opt/app/
# COPY requirements.txt /opt/app/
# If you have additional modules or packages, copy them explicitly or
# uncomment the following line to copy the repo (but ensure .dockerignore is configured)
# COPY . /opt/app
COPY .env /opt/app/
COPY app.py /opt/app/
# # Create a non-root user for running the app
# RUN groupadd -r app && useradd -r -g app app \
#  && chown -R app:app /opt/app

EXPOSE 7860 8000 9100

# USER app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/opt/app/app.py"]