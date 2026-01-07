# LangChain RAG Learning Project

A comprehensive learning platform for LangChain and RAG (Retrieval-Augmented Generation) technologies, providing both theoretical knowledge and practical implementation experience.

## 🎯 Project Overview

This project offers a complete learning journey for developers interested in:
- Understanding LangChain framework fundamentals
- Implementing RAG (Retrieval-Augmented Generation) systems
- Building production-ready LLM applications
- Learning best practices for vector databases and embeddings
- Exploring advanced RAG techniques and optimizations

## 🚀 Features

- **Interactive Learning Modules**: Step-by-step tutorials with theory and practice
- **Complete RAG Implementation**: Full-featured RAG system with multiple retrieval strategies
- **Multi-LLM Support**: Integration with OpenAI, Anthropic, HuggingFace, and local models
- **Vector Database Integration**: Support for Chroma, FAISS, and Qdrant
- **Web Interface**: Streamlit-based UI for easy interaction and testing
- **REST API**: FastAPI-based backend for programmatic access
- **Evaluation Framework**: Built-in metrics and benchmarking tools
- **Production Ready**: Docker deployment with monitoring and logging

## 📋 Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

## 🛠️ Installation

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/your-username/langchain-rag-learning.git
cd langchain-rag-learning

# Copy environment configuration
cp .env.example .env

# Edit .env file with your API keys and configuration
# nano .env

# Start the application with Docker Compose
docker-compose up -d
```

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/langchain-rag-learning.git
cd langchain-rag-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Setup development environment
make setup

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
make run-api  # Start FastAPI server
make run-ui   # Start Streamlit UI (in another terminal)
```

## 🔧 Configuration

The application uses environment variables for configuration. Key settings include:

- **LLM API Keys**: OpenAI, Anthropic, HuggingFace tokens
- **Database URLs**: PostgreSQL and Redis connection strings
- **Vector Database**: Chroma and FAISS configuration
- **File Storage**: Upload directory and size limits
- **RAG Parameters**: Chunk size, retrieval settings, model parameters

See `.env.example` for all available configuration options.

## 📚 Usage

### Web Interface

1. Start the application: `docker-compose up -d`
2. Open your browser to:
   - **Streamlit UI**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs

### API Usage

```python
import httpx

# Upload a document
with open("document.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": f}
    )

# Query the RAG system
response = httpx.post(
    "http://localhost:8000/api/v1/query",
    json={"question": "What is RAG?", "kb_id": "default"}
)
print(response.json())
```

### Learning Path

1. **Theory Modules**: Start with LangChain and RAG fundamentals
2. **Basic Implementation**: Build your first RAG system
3. **Advanced Features**: Explore hybrid retrieval and reranking
4. **Optimization**: Learn performance tuning and evaluation
5. **Production**: Deploy and monitor your RAG application

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Run with coverage
pytest --cov=src/langchain_rag_learning --cov-report=html
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Learning      │
│                 │────│   Backend       │────│   Modules       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   RAG Engine    │
                       └─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Store   │    │   LLM Manager   │    │   Document      │
│  (Chroma/FAISS) │    │   (Multi-LLM)   │    │   Processor     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `make test`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📖 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [User Guide](docs/user-guide.md) - Comprehensive usage guide
- [Developer Guide](docs/developer-guide.md) - Development setup and guidelines
- [Architecture Guide](docs/architecture.md) - System design and components

## 🔒 Security

Please report security vulnerabilities to [security@example.com](mailto:security@example.com). See [SECURITY.md](SECURITY.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - The amazing LLM framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for APIs
- [Streamlit](https://streamlit.io/) - Rapid web app development
- [Chroma](https://www.trychroma.com/) - Vector database for embeddings

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 📖 Documentation: [docs.example.com](https://docs.example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/langchain-rag-learning/issues)

---

**Happy Learning! 🚀**