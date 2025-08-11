#!/bin/bash
set -e

# WASM-Torch Production Entrypoint Script
# Handles initialization, health checks, and service startup

# Environment setup
export PYTHONPATH=${PYTHONPATH:-/app/src}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export WORKERS=${WORKERS:-4}
export CONFIG_PATH=${CONFIG_PATH:-/app/config/production.yaml}

# Logging function
log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")] [ENTRYPOINT] $1" >&2
}

log "Starting WASM-Torch production container"

# Check if running as root (security check)
if [ "$(id -u)" = "0" ]; then
    log "WARNING: Running as root user is not recommended"
fi

# Validate required environment variables
if [ -z "$CONFIG_PATH" ]; then
    log "ERROR: CONFIG_PATH environment variable not set"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    log "ERROR: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

# Create required directories
mkdir -p /app/cache /app/logs /tmp/prometheus_multiproc
chmod 755 /app/cache /app/logs
chmod 777 /tmp/prometheus_multiproc

# Initialize Emscripten environment
if [ -d "/opt/emsdk" ]; then
    log "Initializing Emscripten environment"
    source /opt/emsdk/emsdk_env.sh > /dev/null 2>&1
    log "Emscripten initialized: $(emcc --version | head -n1)"
else
    log "WARNING: Emscripten not found, WASM compilation will be limited"
fi

# System health pre-checks
log "Running system health pre-checks"

# Check memory availability
AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
if [ "$AVAILABLE_MEMORY" -lt 1024 ]; then
    log "WARNING: Low available memory: ${AVAILABLE_MEMORY}MB"
fi

# Check disk space
AVAILABLE_DISK=$(df /app | awk 'NR==2{print $4}')
if [ "$AVAILABLE_DISK" -lt 1048576 ]; then  # 1GB in KB
    log "WARNING: Low available disk space: ${AVAILABLE_DISK}KB"
fi

# Validate Python environment
log "Validating Python environment"
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Validate WASM-Torch installation
if ! python -c "from wasm_torch import __version__; print(f'WASM-Torch version: {__version__}')" 2>/dev/null; then
    log "ERROR: WASM-Torch import failed"
    exit 1
fi

# Command handling
case "$1" in
    serve|server)
        log "Starting WASM-Torch API server with $WORKERS workers"
        exec python -m wasm_torch.server \
            --host 0.0.0.0 \
            --port 8080 \
            --workers "$WORKERS" \
            --config "$CONFIG_PATH" \
            --log-level "$LOG_LEVEL"
        ;;
    
    worker)
        log "Starting WASM-Torch background worker"
        exec python -m wasm_torch.worker \
            --config "$CONFIG_PATH" \
            --log-level "$LOG_LEVEL"
        ;;
    
    health-check|health)
        log "Running health check"
        exec python health-check.py
        ;;
    
    test)
        log "Running comprehensive tests"
        exec python -m pytest tests/ -v --tb=short
        ;;
    
    benchmark)
        log "Running performance benchmarks"
        exec python run_comprehensive_tests.py
        ;;
    
    shell|bash)
        log "Starting interactive shell"
        exec /bin/bash
        ;;
    
    *)
        log "Usage: $0 {serve|worker|health-check|test|benchmark|shell}"
        log "Commands:"
        log "  serve       - Start API server (default)"
        log "  worker      - Start background worker"
        log "  health-check- Run health check"
        log "  test        - Run test suite"
        log "  benchmark   - Run performance benchmarks"
        log "  shell       - Start interactive shell"
        exit 1
        ;;
esac