# syntax=docker/dockerfile:experimental
########################################
# Multi-stage, multi-platform Dockerfile
# - builder: installs Python deps into /install and validates model file
# - runtime: small final image, copies only runtime files
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

# Copy the model into the builder so we can verify it is the real binary
# (this will fail early if it's an LFS pointer text file)
COPY model_100.pth /build/model_100.pth
RUN python - <<'PY'
        import sys, os
        p = "/build/model_100.pth"
        if not os.path.exists(p):
            print("ERROR: model_100.pth not found in build context")
            sys.exit(1)
        with open(p, "rb") as f:
            head = f.read(128)
        # Git LFS pointer files start with: "version https://git-lfs.github.com/spec/v1"
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            print("ERROR: model_100.pth appears to be a Git LFS pointer file, not the real model.")
            print("Please ensure Git LFS is installed and that the real file is present in the build context.")
            sys.exit(2)
        else:
            # print size for diagnostic logs (ok to print numeric size, don't print content)
            size = os.path.getsize(p)
            print(f"model_100.pth looks like a binary file, size={size} bytes")
PY

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

# Copy validated model from builder
COPY --from=builder /build/model_100.pth /opt/app/

# Copy remaining application files
COPY example1.png /opt/app/
COPY example2.png /opt/app/
COPY example3.png /opt/app/
COPY app.py /opt/app/

EXPOSE 7860 8000 9100

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/opt/app/app.py"]
