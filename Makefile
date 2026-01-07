# LangChain RAG Learning Project Makefile

.PHONY: help setup-env activate install install-dev setup clean test test-unit test-integration lint format type-check security pre-commit run-api run-ui build docker-build docker-run docs

# Default target
help:
	@echo "LangChain RAG Learning Project"
	@echo "=============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup-env          Set up conda environment and initial configuration"
	@echo "  activate           Show command to activate conda environment"
	@echo "  install            Install dependencies in conda environment"
	@echo "  install-dev        Install development dependencies"
	@echo "  setup              Complete development setup"
	@echo ""
	@echo "Development Commands:"
	@echo "  run-api            Start FastAPI server"
	@echo "  run-ui             Start Streamlit UI"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  lint               Run linting checks"
	@echo "  format             Format code with black and isort"
	@echo "  type-check         Run type checking with mypy"
	@echo "  security           Run security checks"
	@echo "  pre-commit         Run all pre-commit checks"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build       Build Docker image"
	@echo "  docker-run         Start services with docker-compose"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean              Clean up temporary files"
	@echo "  check-config       Check configuration"
	@echo "  quickstart         Complete setup and start"

# Environment setup
setup-env:
	@echo "Setting up conda environment..."
	python setup_environment.py

activate:
	@echo "To activate the conda environment, run:"
	@echo "conda activate langchain-rag-learning"

# Installation
install:
	conda run -n langchain-rag-learning pip install -e .

install-dev:
	conda run -n langchain-rag-learning pip install -e ".[dev,test,docs]"
	conda run -n langchain-rag-learning pre-commit install

setup: install-dev
	conda run -n langchain-rag-learning pre-commit install --hook-type commit-msg
	mkdir -p data/uploads data/chroma_db data/faiss_index logs config
	@echo "Development environment setup complete!"

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Testing
test:
	conda run -n langchain-rag-learning pytest tests/ -v --cov=src/langchain_rag_learning --cov-report=html --cov-report=term

test-unit:
	conda run -n langchain-rag-learning pytest tests/unit/ -v -m "not integration"

test-integration:
	conda run -n langchain-rag-learning pytest tests/integration/ -v -m integration

# Code quality
lint:
	conda run -n langchain-rag-learning flake8 src/ tests/
	conda run -n langchain-rag-learning bandit -r src/ -f json -o bandit-report.json || true

format:
	conda run -n langchain-rag-learning black src/ tests/ scripts/
	conda run -n langchain-rag-learning isort src/ tests/ scripts/

type-check:
	conda run -n langchain-rag-learning mypy src/

security:
	conda run -n langchain-rag-learning bandit -r src/
	conda run -n langchain-rag-learning safety check

pre-commit:
	conda run -n langchain-rag-learning pre-commit run --all-files

# Development servers
run-api:
	@echo "Starting FastAPI server..."
	conda run -n langchain-rag-learning uvicorn src.langchain_rag_learning.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	@echo "Starting Streamlit UI..."
	conda run -n langchain-rag-learning streamlit run src/langchain_rag_learning/ui/main.py --server.port 8501

# Building
build:
	conda run -n langchain-rag-learning python -m build

# Docker
docker-build:
	docker build -t langchain-rag-learning .

docker-run:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Documentation
docs:
	conda run -n langchain-rag-learning mkdocs build

docs-serve:
	conda run -n langchain-rag-learning mkdocs serve

# Database
db-upgrade:
	conda run -n langchain-rag-learning alembic upgrade head

db-downgrade:
	conda run -n langchain-rag-learning alembic downgrade -1

db-revision:
	conda run -n langchain-rag-learning alembic revision --autogenerate -m "$(message)"

# Configuration
check-config:
	@echo "Checking configuration..."
	conda run -n langchain-rag-learning python -c "from src.langchain_rag_learning.core.config import get_llm_config, get_settings; config = get_llm_config(); print('Available providers:', config.get_enabled_providers()); print('Default provider:', config.get_default_provider())"

# Ollama local model management
ollama-install:
	@echo "Please install Ollama from https://ollama.ai/"
	@echo "Then run: make ollama-pull-models"

ollama-pull-models:
	ollama pull llama2
	ollama pull codellama
	ollama pull mistral

ollama-list:
	ollama list

# Utilities
check-deps:
	conda run -n langchain-rag-learning pip-audit

update-deps:
	conda run -n langchain-rag-learning pip-compile requirements.in
	conda run -n langchain-rag-learning pip-compile requirements-dev.in

# CI/CD helpers
ci-test: lint type-check security test

ci-build: clean build

# Development helpers
dev-setup: setup
	@echo "Creating sample data directories..."
	mkdir -p data/sample_docs data/knowledge_bases data/benchmarks
	@echo "Development setup complete!"

reset-db:
	rm -f data/rag_learning.db
	conda run -n langchain-rag-learning alembic upgrade head

logs:
	tail -f logs/app.log

# Quick start
quickstart: setup-env
	@echo ""
	@echo "🎉 Quick start completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. conda activate langchain-rag-learning"
	@echo "2. Edit .env file with your API keys"
	@echo "3. make run-api (in one terminal)"
	@echo "4. make run-ui (in another terminal)"
	@echo ""
	@echo "💡 Recommended: Get a DeepSeek API key (very affordable)"
	@echo "   Visit: https://platform.deepseek.com/"