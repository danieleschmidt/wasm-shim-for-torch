#!/bin/bash

# Development environment setup script
# This script runs after the devcontainer is created

set -e

echo "🚀 Setting up WASM Torch development environment..."

# Ensure we're in the workspace directory
cd /workspace

# Install project in development mode
echo "📦 Installing project in development mode..."
pip install -e .[dev,test,docs,build]

# Initialize Emscripten environment
echo "🔧 Initializing Emscripten environment..."
if [ -d "/home/vscode/emsdk" ]; then
    source /home/vscode/emsdk/emsdk_env.sh
    echo "✅ Emscripten environment initialized"
else
    echo "⚠️  Emscripten not found, skipping initialization"
fi

# Verify Rust and WASM tools
echo "🦀 Verifying Rust and WASM tools..."
if command -v cargo &> /dev/null; then
    cargo --version
    rustc --version
    echo "✅ Rust toolchain ready"
else
    echo "⚠️  Rust not found"
fi

# Verify Wasmtime
echo "⚡ Verifying Wasmtime runtime..."
if command -v wasmtime &> /dev/null; then
    wasmtime --version
    echo "✅ Wasmtime runtime ready"
else
    echo "⚠️  Wasmtime not found"
fi

# Install pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  Pre-commit config not found"
fi

# Setup Git hooks for security
echo "🛡️  Setting up security Git hooks..."
mkdir -p .git/hooks

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Security pre-commit hook

echo "🔍 Running security checks..."

# Check for secrets
if command -v git-secrets &> /dev/null; then
    git secrets --scan
fi

# Check for large files
find . -type f -size +50M -not -path './.git/*' -not -path './node_modules/*' -not -path './.venv/*' | while read file; do
    echo "⚠️  Large file detected: $file"
done

# Run bandit security linter
if command -v bandit &> /dev/null; then
    bandit -r src/ -ll -f json -o bandit-report.json || true
fi

echo "✅ Security checks completed"
EOF

chmod +x .git/hooks/pre-commit

# Setup development certificates for HTTPS testing
echo "🔐 Setting up development certificates..."
mkdir -p certs/
if command -v openssl &> /dev/null; then
    openssl req -x509 -newkey rsa:4096 -keyout certs/dev-key.pem -out certs/dev-cert.pem -days 365 -nodes -subj "/C=US/ST=Development/L=Local/O=WASM-Torch/CN=localhost"
    echo "✅ Development certificates created"
fi

# Create common development directories
echo "📁 Creating development directories..."
mkdir -p {logs,tmp,cache,uploads,models,artifacts}
mkdir -p tests/{unit,integration,performance,security,browser}
mkdir -p docs/{api,tutorials,examples}
mkdir -p scripts/{build,deploy,maintenance}

# Setup environment variables template
echo "⚙️  Creating environment template..."
cat > .env.template << 'EOF'
# Development Environment Variables
# Copy this file to .env and customize for your setup

# Application
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://dev:dev_password@postgres:5432/wasm_torch_dev

# Redis
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
JAEGER_URL=http://jaeger:16686

# Object Storage
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# WASM Configuration
WASM_MEMORY_LIMIT=256MB
WASM_STACK_SIZE=1MB
EMSCRIPTEN_CACHE_DIR=/tmp/emscripten_cache

# Testing
TEST_DATABASE_URL=postgresql://dev:dev_password@postgres:5432/wasm_torch_test
BROWSER_TIMEOUT=30000
HEADLESS_BROWSER=true

# AI/ML
MODEL_CACHE_DIR=/workspace/cache/models
MAX_MODEL_SIZE=100MB
INFERENCE_TIMEOUT=30
EOF

# Initialize test data
echo "🧪 Setting up test environment..."
if [ -f "scripts/generate-test-data.py" ]; then
    python scripts/generate-test-data.py
    echo "✅ Test data generated"
fi

# Build initial WASM modules for testing
echo "🏗️  Building initial WASM modules..."
if [ -f "scripts/build-test-modules.sh" ]; then
    bash scripts/build-test-modules.sh
    echo "✅ Test WASM modules built"
fi

# Setup VS Code workspace settings
echo "⚙️  Configuring VS Code workspace..."
mkdir -p .vscode

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--no-cov"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: Debug Application",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/wasm_torch/cli.py",
            "args": ["--debug"],
            "console": "integratedTerminal",
            "env": {
                "DEBUG": "true",
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "WASM: Debug Module", 
            "type": "node",
            "request": "launch",
            "program": "${workspaceFolder}/tests/browser/debug-wasm.js",
            "console": "integratedTerminal"
        }
    ]
}
EOF

cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build WASM Module",
            "type": "shell",
            "command": "python",
            "args": [
                "-m", "wasm_torch.build",
                "--input", "${input:modelPath}",
                "--output", "${workspaceFolder}/build/",
                "--optimize"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Tests",
            "type": "shell", 
            "command": "pytest",
            "args": [
                "tests/",
                "-v",
                "--cov=wasm_torch",
                "--cov-report=html"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Security Scan",
            "type": "shell",
            "command": "bash",
            "args": [
                "scripts/security-scan.sh"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Start Development Server",
            "type": "shell",
            "command": "python",
            "args": [
                "-m", "uvicorn",
                "wasm_torch.server:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8080"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ],
    "inputs": [
        {
            "id": "modelPath",
            "description": "Path to PyTorch model file",
            "default": "models/example.pth",
            "type": "promptString"
        }
    ]
}
EOF

# Setup Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name wasm-torch --display-name "WASM Torch"

# Create example notebooks
mkdir -p notebooks
cat > notebooks/getting-started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WASM Torch Getting Started\n",
    "\n",
    "This notebook demonstrates the basic usage of WASM Torch for converting PyTorch models to WebAssembly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from wasm_torch import export_to_wasm, WASMRuntime\n",
    "\n",
    "# Create a simple model\n",
    "model = nn.Sequential(\n",
    "    nn.Linear(784, 128),\n",
    "    nn.ReLU(),\n",
    "    nn.Linear(128, 10)\n",
    ")\n",
    "\n",
    "# Export to WASM\n",
    "export_to_wasm(\n",
    "    model,\n",
    "    'simple_model.wasm',\n",
    "    example_input=torch.randn(1, 784)\n",
    ")\n",
    "\n",
    "print('✅ Model exported to WASM successfully!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "WASM Torch",
   "language": "python", 
   "name": "wasm-torch"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Setup monitoring dashboards
echo "📊 Setting up monitoring dashboards..."
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}

cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'wasm-torch-app'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
EOF

# Create development database
echo "🗄️  Setting up development database..."
if command -v psql &> /dev/null; then
    export PGPASSWORD=dev_password
    createdb -h postgres -U dev wasm_torch_dev 2>/dev/null || echo "Database already exists"
    createdb -h postgres -U dev wasm_torch_test 2>/dev/null || echo "Test database already exists"
    echo "✅ Development databases ready"
fi

# Final setup verification
echo "🔍 Verifying development environment..."

# Check Python installation
python --version
pip --version

# Check key packages
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch not found"
python -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null || echo "⚠️  NumPy not found"

# Check WASM tools
if command -v wasm-opt &> /dev/null; then
    wasm-opt --version | head -1
    echo "✅ WASM optimization tools ready"
fi

# Check development tools
if command -v pytest &> /dev/null; then
    pytest --version
    echo "✅ Testing framework ready"
fi

if command -v black &> /dev/null; then
    black --version
    echo "✅ Code formatting tools ready"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Quick start commands:"
echo "  • Run tests: pytest tests/"
echo "  • Format code: black src/ tests/"
echo "  • Lint code: ruff check src/ tests/"
echo "  • Start server: python -m uvicorn wasm_torch.server:app --reload"
echo "  • Open Jupyter: jupyter lab --ip 0.0.0.0 --port 8888"
echo "  • View logs: docker-compose logs -f"
echo ""
echo "🔧 Available services:"
echo "  • Grafana: http://localhost:3001 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • Jaeger: http://localhost:16686"
echo "  • Kibana: http://localhost:5601"
echo "  • MinIO: http://localhost:9001 (minioadmin/minioadmin123)"
echo ""
echo "Happy coding! 🚀"