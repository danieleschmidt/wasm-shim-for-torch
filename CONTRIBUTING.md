# Contributing to WASM Shim for Torch

We welcome contributions to the WASM Shim for Torch project! This guide will help you get started with contributing to our WebAssembly-powered PyTorch inference library.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/wasm-shim-for-torch.git
   cd wasm-shim-for-torch
   ```
3. **Set up development environment**:
   ```bash
   make dev-setup
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Environment

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for JavaScript/TypeScript components)
- CMake 3.26+
- Ninja build system
- Git

### Setup

```bash
# Complete development setup
make dev-setup

# Or manual setup:
pip install -e ".[dev,test,docs,build]"
pre-commit install
make setup-emscripten
```

### Verification

```bash
# Run tests to verify setup
make test

# Run linting
make lint

# Build WASM components (optional)
make wasm
```

## 📋 Development Workflow

### Before Making Changes

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for new features or major changes
3. **Discuss the approach** with maintainers before implementing

### Making Changes

1. **Write tests first** (TDD approach preferred)
2. **Follow code style** guidelines (enforced by pre-commit hooks)
3. **Keep changes focused** - one feature/fix per PR
4. **Update documentation** as needed

### Code Style

We use automated code formatting and linting:

- **Black** for Python code formatting
- **Ruff** for linting and import sorting  
- **MyPy** for type checking
- **Pre-commit hooks** enforce style automatically

```bash
# Format code
make format

# Check style
make lint

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run fast tests only
make test-fast

# Run specific test file
pytest tests/test_export.py -v
```

### Documentation

```bash
# Build documentation locally
make docs

# Serve documentation
make docs-serve
```

## 🎯 Contribution Areas

### High Priority

1. **Core WASM Runtime Implementation**
   - WASI-NN interface implementation
   - Memory management optimization
   - SIMD operation kernels

2. **PyTorch Operator Support**
   - Additional layer implementations
   - Custom operator registration
   - Quantization support

3. **Performance Optimization**
   - SIMD optimizations
   - Memory layout improvements
   - Threading optimizations

4. **Browser Compatibility**
   - Cross-browser testing
   - Mobile browser support
   - WebNN integration

### Medium Priority

1. **Developer Experience**
   - CLI improvements
   - Better error messages
   - Debugging tools

2. **Documentation**
   - Tutorial content
   - API documentation
   - Example applications

3. **Testing**
   - Browser automation tests
   - Performance benchmarks
   - Integration tests

### Getting Started Areas

1. **Examples and Demos**
   - Model conversion examples
   - Browser demo applications
   - Performance comparisons

2. **Documentation**
   - Improving existing docs
   - Adding code comments
   - Creating tutorials

3. **Testing**
   - Writing test cases
   - Improving test coverage
   - Adding edge case tests

## 📝 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   make test
   make lint
   ```

2. **Update documentation** if needed
3. **Add/update tests** for new functionality
4. **Check that CI will pass** locally

### PR Guidelines

1. **Clear title** describing the change
2. **Detailed description** explaining:
   - What changes were made
   - Why they were necessary
   - How to test them
3. **Link related issues** using keywords (fixes #123)
4. **Small, focused changes** - large PRs are hard to review

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Tested in browser environment (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** in multiple environments
4. **Approval** before merging

## 🔧 Technical Guidelines

### Code Organization

```
src/wasm_torch/
├── __init__.py          # Package exports
├── export.py            # Model export functionality
├── runtime.py           # WASM runtime implementation
├── optimize.py          # Optimization utilities
└── cli.py              # Command-line interface
```

### Architecture Principles

1. **Modular design** - clear separation of concerns
2. **Type safety** - comprehensive type hints
3. **Error handling** - informative error messages
4. **Performance** - optimize for browser constraints
5. **Compatibility** - support multiple PyTorch versions

### API Design

1. **Pythonic APIs** following PyTorch conventions
2. **Async/await** for browser operations
3. **Clear abstractions** hiding WASM complexity
4. **Backward compatibility** when possible

### Security Considerations

1. **Input validation** for all user data
2. **Safe WASM execution** in browser sandbox
3. **No arbitrary code execution**
4. **Secure defaults** for all configurations

## 🐛 Bug Reports

### Before Reporting

1. **Check existing issues** for duplicates
2. **Try latest version** - bug might be fixed
3. **Minimal reproduction** - simplify the problem

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Environment
- OS: [e.g., macOS 12.0]
- Browser: [e.g., Chrome 126]
- Python: [e.g., 3.11.5]
- PyTorch: [e.g., 2.4.0]
- wasm-torch: [e.g., 0.1.0]

## Reproduction Steps
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Additional Context
Logs, screenshots, etc.
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing issues** and roadmap
2. **Consider scope** - fits project goals?
3. **Implementation complexity** - is it feasible?

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should this be implemented?

## Alternatives Considered
Other approaches considered

## Additional Context
Examples, use cases, etc.
```

## 📚 Resources

### Documentation
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](https://wasm-torch.readthedocs.io)
- [Browser Compatibility](docs/BROWSER_COMPAT.md)

### External Resources
- [WebAssembly Specification](https://webassembly.github.io/spec/)
- [WASI-NN Specification](https://github.com/WebAssembly/wasi-nn)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)

### Community
- [GitHub Discussions](https://github.com/yourusername/wasm-shim-for-torch/discussions)
- [Discord Server](https://discord.gg/wasm-torch)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/wasm-torch)

## 🎖️ Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributor graph**
- **Special mentions** in documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) for details.

## ❓ Questions

- **General questions**: [GitHub Discussions](https://github.com/yourusername/wasm-shim-for-torch/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/yourusername/wasm-shim-for-torch/issues)
- **Security issues**: Email security@yourdomain.com
- **Direct contact**: Email maintainers@yourdomain.com

Thank you for contributing to WASM Shim for Torch! 🚀