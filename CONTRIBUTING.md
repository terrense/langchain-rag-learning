# Contributing to LangChain RAG Learning Project

Thank you for your interest in contributing to the LangChain RAG Learning Project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when creating new issues
3. **Provide detailed information** including:
   - Environment details (OS, Python version, etc.)
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs or screenshots

### Suggesting Features

1. **Check the roadmap** to see if the feature is already planned
2. **Open a feature request** using the template
3. **Discuss the feature** with maintainers before implementation
4. **Consider the scope** - keep features focused and well-defined

### Code Contributions

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** (see below)
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for full stack testing)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/langchain-rag-learning.git
cd langchain-rag-learning

# Set up development environment
make setup

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Run with coverage
pytest --cov=src/langchain_rag_learning --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security

# Run all quality checks
make pre-commit
```

## 📝 Coding Standards

### Python Style

- **Follow PEP 8** with line length of 88 characters
- **Use Black** for code formatting
- **Use isort** for import sorting
- **Use type hints** for all function signatures
- **Write docstrings** for all public functions and classes

### Code Organization

```python
"""Module docstring describing the purpose."""

import standard_library
import third_party_library

from langchain_rag_learning import local_module


class ExampleClass:
    """Class docstring with clear description.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    """
    
    def __init__(self, param1: str, param2: int) -> None:
        self.param1 = param1
        self.param2 = param2
    
    def public_method(self, arg: str) -> str:
        """Public method with clear docstring.
        
        Args:
            arg: Description of argument
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: When arg is invalid
        """
        return self._private_method(arg)
    
    def _private_method(self, arg: str) -> str:
        """Private method for internal use."""
        return f"processed_{arg}"
```

### Testing Guidelines

- **Write tests first** (TDD approach encouraged)
- **Test public interfaces** thoroughly
- **Use descriptive test names** that explain what is being tested
- **Mock external dependencies** in unit tests
- **Use fixtures** for common test data

```python
import pytest
from unittest.mock import Mock, patch

from langchain_rag_learning.core import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    @pytest.fixture
    def example_instance(self):
        """Create an instance for testing."""
        return ExampleClass("test", 42)
    
    def test_public_method_returns_expected_value(self, example_instance):
        """Test that public_method returns the expected processed value."""
        result = example_instance.public_method("input")
        assert result == "processed_input"
    
    def test_public_method_raises_on_invalid_input(self, example_instance):
        """Test that public_method raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            example_instance.public_method("")
    
    @patch('langchain_rag_learning.core.external_service')
    def test_method_with_external_dependency(self, mock_service, example_instance):
        """Test method that depends on external service."""
        mock_service.return_value = "mocked_response"
        result = example_instance.method_with_dependency()
        assert result == "expected_result"
        mock_service.assert_called_once()
```

## 📋 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**: `make test`
2. **Run code quality checks**: `make pre-commit`
3. **Update documentation** if needed
4. **Add changelog entry** for significant changes
5. **Rebase on latest main** branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** in development environment
4. **Approval** and merge by maintainer

## 🏗️ Architecture Guidelines

### Module Structure

```
src/langchain_rag_learning/
├── core/           # Core utilities and configuration
├── learning/       # Learning modules and content
├── rag/           # RAG engine components
├── api/           # FastAPI application
├── ui/            # Streamlit interface
└── utils/         # Shared utilities
```

### Design Principles

- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Injection**: Use dependency injection for testability
- **Interface Segregation**: Define clear interfaces between components
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: Structured logging throughout the application

### Adding New Features

1. **Design the interface** first
2. **Write tests** for the interface
3. **Implement the feature** following existing patterns
4. **Add documentation** and examples
5. **Update configuration** if needed

## 📚 Documentation

### Types of Documentation

- **API Documentation**: Auto-generated from docstrings
- **User Guides**: Step-by-step tutorials
- **Developer Guides**: Technical implementation details
- **Architecture Docs**: System design and decisions

### Writing Guidelines

- **Use clear, concise language**
- **Include code examples** where helpful
- **Keep documentation up-to-date** with code changes
- **Use consistent formatting** and structure

## 🐛 Debugging

### Common Issues

1. **Import errors**: Check Python path and virtual environment
2. **API key issues**: Verify environment variables are set
3. **Database connection**: Ensure services are running
4. **Vector store errors**: Check data directory permissions

### Debugging Tools

- **Logging**: Use structured logging for debugging
- **Debugger**: Use `pdb` or IDE debugger for step-through debugging
- **Profiling**: Use `py-spy` for performance profiling
- **Testing**: Write focused tests to isolate issues

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite** and quality checks
4. **Create release PR** and get approval
5. **Tag release** and push to main
6. **Automated deployment** via GitHub Actions

## 💬 Communication

### Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and community support
- **Email**: Security issues and private matters

### Guidelines

- **Be respectful** and inclusive
- **Search before asking** to avoid duplicates
- **Provide context** when asking questions
- **Help others** when you can

## 🎯 Roadmap

### Current Focus

- Core RAG functionality
- Learning module content
- API stability
- Documentation completeness

### Future Plans

- Advanced RAG techniques
- Multi-modal support
- Cloud deployment options
- Performance optimizations

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the LangChain RAG Learning Project! 🙏