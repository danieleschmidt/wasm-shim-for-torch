.PHONY: help install install-dev clean lint test test-cov docs build wasm setup-emscripten

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package in development mode with all dependencies
	pip install -e ".[dev,test,docs,build]"
	pre-commit install

clean:  ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting tools
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:  ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=wasm_torch --cov-report=html --cov-report=term

test-fast:  ## Run tests excluding slow ones
	pytest -m "not slow"

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build:  ## Build Python package
	python -m build

setup-emscripten:  ## Setup Emscripten toolchain
	@if [ ! -d "emsdk" ]; then \
		git clone https://github.com/emscripten-core/emsdk.git; \
	fi
	cd emsdk && ./emsdk install latest && ./emsdk activate latest

wasm: setup-emscripten  ## Build WASM components
	@echo "Building WASM components..."
	@if [ ! -d "build" ]; then mkdir build; fi
	cd build && \
	source ../emsdk/emsdk_env.sh && \
	emcmake cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DUSE_SIMD=ON \
		-DUSE_THREADS=ON \
		-DWASM_TORCH_VERSION=0.1.0 && \
	emmake make -j$$(nproc)

benchmark:  ## Run performance benchmarks
	pytest tests/benchmarks/ -v

security:  ## Run security checks
	bandit -r src/
	safety check

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

check:  ## Run all checks (lint, test, security)
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security

release:  ## Build and upload to PyPI (requires authentication)
	$(MAKE) clean
	$(MAKE) build
	twine check dist/*
	twine upload dist/*

dev-setup:  ## Complete development environment setup
	$(MAKE) install-dev
	$(MAKE) setup-emscripten
	@echo "Development environment ready!"