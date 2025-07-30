# Development container for WASM Torch project
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

# Build arguments
ARG INSTALL_NODE="true"
ARG INSTALL_CMAKE="true"  
ARG INSTALL_EMSCRIPTEN="true"
ARG EMSCRIPTEN_VERSION="3.1.61"
ARG WASMTIME_VERSION="10.0.0"

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        gdb \
        lcov \
        pkg-config \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libgdbm-compat-dev \
        liblzma-dev \
        libncurses5-dev \
        libreadline6-dev \
        libsqlite3-dev \
        libssl-dev \
        lzma \
        lzma-dev \
        tk-dev \
        uuid-dev \
        zlib1g-dev \
        curl \
        wget \
        git \
        vim \
        nano \
        htop \
        tree \
        jq \
        unzip \
        zip

# Install Node.js if requested
RUN if [ "${INSTALL_NODE}" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - \
        && apt-get install -y nodejs \
        && npm install -g yarn typescript @types/node; \
    fi

# Install CMake if requested  
RUN if [ "${INSTALL_CMAKE}" = "true" ]; then \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
        && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
        && apt-get update \
        && apt-get install -y cmake ninja-build; \
    fi

# Switch to vscode user for remaining operations
USER vscode

# Install Python development tools
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        pip-tools \
        pre-commit \
        black \
        ruff \
        mypy \
        pytest \
        pytest-cov \
        pytest-benchmark \
        pytest-xdist \
        hypothesis \
        bandit[toml] \
        safety \
        pip-audit \
        build \
        twine \
        ipython \
        jupyter \
        notebook \
        jupyterlab

# Install Emscripten SDK if requested
RUN if [ "${INSTALL_EMSCRIPTEN}" = "true" ]; then \
        cd /home/vscode \
        && git clone https://github.com/emscripten-core/emsdk.git \
        && cd emsdk \
        && ./emsdk install ${EMSCRIPTEN_VERSION} \
        && ./emsdk activate ${EMSCRIPTEN_VERSION}; \
    fi

# Install Wasmtime runtime
RUN cd /home/vscode \
    && curl -sSf https://wasmtime.dev/install.sh | bash -s -- --version v${WASMTIME_VERSION}

# Install Rust for WASM tooling
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && /home/vscode/.cargo/bin/rustup target add wasm32-unknown-unknown \
    && /home/vscode/.cargo/bin/rustup target add wasm32-wasi \
    && /home/vscode/.cargo/bin/cargo install wasm-pack

# Install additional WASM tools
RUN /home/vscode/.cargo/bin/cargo install wabt wasm-opt binaryen

# Install PyTorch with WASM support dependencies
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        torch>=2.4.0 \
        torchvision \
        torchaudio \
        numpy>=1.24.0 \
        pybind11>=2.11.0 \
        typing-extensions>=4.0.0

# Install browser automation tools for testing
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        playwright \
        selenium \
    && playwright install chromium firefox webkit

# Install monitoring and observability tools
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-instrumentation \
        opentelemetry-exporter-prometheus \
        opentelemetry-exporter-jaeger \
        prometheus-client \
        psutil \
        memory-profiler

# Install security scanning tools
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        semgrep \
        cyclonedx-bom \
        syft

# Install development utilities
RUN pip3 --disable-pip-version-check --no-cache-dir install \
        httpx \
        fastapi \
        uvicorn \
        aiofiles \
        python-multipart \
        pydantic \
        sqlalchemy \
        alembic \
        redis \
        celery

# Configure shell environment
RUN echo 'export PATH="/home/vscode/.cargo/bin:/home/vscode/.wasmtime/bin:$PATH"' >> /home/vscode/.bashrc \
    && echo 'export PATH="/home/vscode/emsdk:$PATH"' >> /home/vscode/.bashrc \
    && echo 'export EMSDK="/home/vscode/emsdk"' >> /home/vscode/.bashrc \
    && echo 'source /home/vscode/emsdk/emsdk_env.sh' >> /home/vscode/.bashrc

# Configure Git (will be overridden by user's Git config)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.autocrlf input

# Create workspace directory structure
RUN mkdir -p /home/vscode/.vscode-server/extensions \
    && mkdir -p /home/vscode/workspace/{src,tests,docs,scripts,build,dist}

# Set working directory
WORKDIR /workspace

# Install development server for local testing
COPY --chown=vscode:vscode package.json* ./
RUN if [ -f "package.json" ]; then npm install; fi

# Copy and install Python dependencies
COPY --chown=vscode:vscode requirements*.txt ./
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi \
    && if [ -f "requirements-dev.txt" ]; then pip install -r requirements-dev.txt; fi

# Setup pre-commit hooks
COPY --chown=vscode:vscode .pre-commit-config.yaml ./
RUN if [ -f ".pre-commit-config.yaml" ]; then pre-commit install; fi

# Expose common development ports
EXPOSE 3000 8080 9090 3001 16686 5601

# Keep container running
CMD ["sleep", "infinity"]