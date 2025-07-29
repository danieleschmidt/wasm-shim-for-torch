# Multi-stage Dockerfile for WASM Shim for Torch
# Supports both development and production builds

ARG PYTHON_VERSION=3.11
ARG EMSCRIPTEN_VERSION=3.1.61

# Base Python image
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

WORKDIR /workspace

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install Emscripten
ARG EMSCRIPTEN_VERSION
RUN git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk && \
    cd /opt/emsdk && \
    ./emsdk install ${EMSCRIPTEN_VERSION} && \
    ./emsdk activate ${EMSCRIPTEN_VERSION}

ENV PATH="/opt/emsdk:/opt/emsdk/upstream/emscripten:${PATH}" \
    EMSDK="/opt/emsdk"

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev,test,docs,build]"

# Default development command
CMD ["bash"]

# Production build stage
FROM base as builder

WORKDIR /build

# Copy requirements and install minimal dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source and build
COPY . .
RUN pip install build && python -m build

# Production runtime stage
FROM python:${PYTHON_VERSION}-slim as production

# Install runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built package and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Create non-root user
RUN useradd --create-home --shell /bin/bash wasm-torch
USER wasm-torch
WORKDIR /home/wasm-torch

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import wasm_torch; print('OK')" || exit 1

# Default production command
CMD ["wasm-torch", "--help"]