# Development Guide

This guide covers the development setup and workflow for WASM Shim for Torch.

## Quick Start

### Prerequisites

- Python 3.10+ with pip
- Git
- Make (for convenience commands)
- Docker (optional, for containerized development)

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/wasm-shim-for-torch.git
   cd wasm-shim-for-torch
   ```

2. **Set up development environment**:
   ```bash
   # Option 1: Direct installation
   make install-dev
   
   # Option 2: Using Docker
   docker-compose up dev
   ```

3. **Verify installation**:
   ```bash
   pytest --version
   wasm-torch --help
   ```

## Development Workflow

### Code Quality

We maintain high code quality through automated tools:

```bash
# Run all quality checks
make check

# Individual tools
make lint      # Run linting (ruff, mypy)
make format    # Auto-format code
make test      # Run test suite
make security  # Security scanning
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest tests/unit/            # Unit tests only
pytest tests/integration/     # Integration tests
pytest tests/benchmarks/      # Performance benchmarks
```

### Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit:

```bash
# Install hooks (done automatically with make install-dev)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

## Architecture Overview

```
src/wasm_torch/
├── __init__.py          # Package initialization and exports
├── cli.py              # Command-line interface
├── export.py           # Model export functionality
├── optimize.py         # Optimization utilities  
└── runtime.py          # WASM runtime interface

tests/
├── conftest.py         # Shared pytest fixtures
├── test_cli.py         # CLI testing
├── test_export.py      # Export functionality tests
├── test_optimize.py    # Optimization tests
├── test_runtime.py     # Runtime tests
├── unit/              # Unit tests
├── integration/       # Integration tests
└── benchmarks/        # Performance benchmarks
```

## WASM Build System

### Emscripten Setup

```bash
# Automatic setup via Makefile
make setup-emscripten

# Manual setup
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Building WASM Components

```bash
# Build all WASM components
make wasm

# Manual build process
mkdir build && cd build
source ../emsdk/emsdk_env.sh
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SIMD=ON \
    -DUSE_THREADS=ON
emmake make -j$(nproc)
```

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual functions and classes
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **Benchmarks** (`tests/benchmarks/`): Performance regression testing
4. **WASM Tests**: Browser-based testing (requires special setup)

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
def test_example(device, small_tensor):
    # device: torch.device (CPU/CUDA)
    # small_tensor: torch.Tensor(2, 3, 4)
    assert small_tensor.device == device
```

### Marking Tests

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_large_model_export():
    # Long-running test
    pass

@pytest.mark.wasm
def test_browser_inference():
    # Requires WASM runtime
    pass

@pytest.mark.benchmark
def test_inference_speed():
    # Performance benchmark
    pass
```

## Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve locally
make docs-serve
# Visit http://localhost:8000
```

### Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst           # Main documentation page
├── api/                # API reference (auto-generated)
├── tutorials/          # Usage tutorials
└── development/        # Development guides
```

## Performance Profiling

### Memory Profiling

```bash
# Install profiling tools
pip install memory-profiler py-spy

# Profile memory usage
python -m memory_profiler examples/profile_export.py

# Profile CPU usage
py-spy record -o profile.svg -- python examples/export_model.py
```

### Benchmark Testing

```bash
# Run performance benchmarks
make benchmark

# Custom benchmark runs
pytest tests/benchmarks/ --benchmark-min-rounds=10
```

## Debugging

### Common Issues

1. **Emscripten Path Issues**:
   ```bash
   # Ensure emsdk is in PATH
   source emsdk/emsdk_env.sh
   which emcc  # Should show emscripten compiler
   ```

2. **PyTorch Version Conflicts**:
   ```bash
   # Check PyTorch version
   python -c "import torch; print(torch.__version__)"
   # Should be >= 2.4.0
   ```

3. **WASM Runtime Errors**:
   ```bash
   # Enable debug mode
   export WASM_TORCH_DEBUG=1
   # Check browser console for detailed errors
   ```

### Debug Builds

```bash
# Build with debug symbols
cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Debug
emmake make -j$(nproc)
```

## Container Development

### Using Docker

```bash
# Start development container
docker-compose up dev

# Run tests in container
docker-compose run test

# Build documentation
docker-compose run docs

# Security scanning
docker-compose run security
```

### Multi-stage Builds

The Dockerfile supports multiple targets:

- `development`: Full development environment with Emscripten
- `production`: Minimal runtime image
- `builder`: Intermediate build stage

## Release Process

### Version Management

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.1`
4. Push tag: `git push origin v0.1.1`

### Building Releases

```bash
# Build distribution packages
make build

# Upload to PyPI (requires authentication)
make release
```

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for CI/CD:

- `ci.yml`: Run tests and quality checks
- `build.yml`: Build packages and WASM components
- `security.yml`: Security scanning
- `docs.yml`: Documentation deployment

### Local CI Testing

```bash
# Test multiple Python versions
tox

# Specific environments
tox -e py311,lint,security
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints for all public APIs
- Write docstrings for public functions
- Keep functions focused and testable

### Commit Messages

Follow conventional commit format:

```
feat: add quantization support for models
fix: resolve memory leak in WASM runtime
docs: update installation instructions
test: add integration tests for export
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run quality checks: `make check`
5. Commit changes with descriptive messages
6. Push to your fork and create a pull request

### Review Checklist

- [ ] Tests pass locally and in CI
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Security scan passes
- [ ] Performance impact assessed
- [ ] Breaking changes documented

## Troubleshooting

### Common Development Issues

1. **Import errors**: Ensure `PYTHONPATH` includes `src/`
2. **Test failures**: Check fixture dependencies and test isolation
3. **Build errors**: Verify Emscripten setup and CMake configuration
4. **Performance regressions**: Run benchmarks and compare results

### Getting Help

- Check existing [GitHub Issues](https://github.com/yourusername/wasm-shim-for-torch/issues)
- Review [Documentation](https://wasm-torch.readthedocs.io)
- Join [Discord Community](https://discord.gg/wasm-torch)

## Development Tools

### Recommended IDE Setup

**VS Code Extensions**:
- Python
- Pylance
- Black Formatter
- isort
- GitLens
- Docker

**PyCharm Configuration**:
- Enable type checking
- Configure code style to match ruff/black
- Set up run configurations for tests

### Shell Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
alias wt-test="pytest tests/"
alias wt-lint="ruff check src/ tests/"
alias wt-fmt="ruff format src/ tests/"
alias wt-build="make wasm"
```

This development guide should help you get started contributing to WASM Shim for Torch. For questions or improvements to this guide, please open an issue or pull request.