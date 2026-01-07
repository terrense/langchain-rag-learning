# 🚀 Quick Start Guide

Get up and running with the LangChain RAG Learning Project in minutes!

## 📋 Prerequisites

- **Conda** (Anaconda or Miniconda) - [Install here](https://docs.conda.io/en/latest/miniconda.html)
- **Git** - For cloning the repository

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd langchain-rag-learning

# Run the automated setup
python setup_environment.py
```

This will:
- ✅ Create conda environment with all dependencies
- ✅ Set up configuration files
- ✅ Create necessary directories

### 2. Activate Environment

```bash
conda activate langchain-rag-learning
```

### 3. Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Recommended: DeepSeek (very affordable - ~$0.0014 per 1K tokens)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Other providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**💡 Get DeepSeek API Key (Recommended):**
- Visit: https://platform.deepseek.com/
- Sign up and get $5 free credits
- Very affordable pricing for learning

### 4. Start the Application

**Terminal 1 - API Server:**
```bash
make run-api
```

**Terminal 2 - Web UI:**
```bash
make run-ui
```

### 5. Access the Application

- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 🆓 Free Options

### Option 1: Local Models (Completely Free)

Install Ollama for local models:

```bash
# Install Ollama from https://ollama.ai/
# Then pull models:
ollama pull llama2
ollama pull codellama
ollama pull mistral

# Check available models
make ollama-list
```

### Option 2: HuggingFace (Free Tier)

Get a free HuggingFace token:
- Visit: https://huggingface.co/settings/tokens
- Add to `.env`: `HUGGINGFACE_API_KEY=your_token_here`

## 🎯 What You Get

### Learning Modules
- **Theory**: LangChain fundamentals, RAG concepts, vector databases
- **Practice**: Step-by-step tutorials with real code
- **Advanced**: Hybrid retrieval, reranking, optimization

### RAG System Features
- **Multi-LLM Support**: OpenAI, Anthropic, DeepSeek, local models
- **Document Processing**: PDF, Word, Markdown, HTML
- **Vector Databases**: Chroma, FAISS integration
- **Web Interface**: Upload docs, chat with your data
- **API Access**: REST API for programmatic use

### Production Features
- **Monitoring**: Performance metrics and logging
- **Testing**: Comprehensive test suite
- **Docker**: Containerized deployment
- **CI/CD**: GitHub Actions workflow

## 🛠️ Development Commands

```bash
# Run tests
make test

# Code formatting
make format

# Type checking
make type-check

# Check configuration
make check-config

# View all commands
make help
```

## 📚 Learning Path

1. **Start with Theory** - Open the web UI and explore learning modules
2. **Upload Documents** - Try the document upload feature
3. **Chat with Data** - Ask questions about your documents
4. **Explore API** - Check out the interactive API docs
5. **Experiment** - Try different models and configurations
6. **Advanced Features** - Explore hybrid retrieval and optimization

## 🔧 Configuration

The system uses a flexible YAML configuration in `config/llm_providers.yaml`:

```yaml
# Easy to modify providers
default_provider: "deepseek"  # Change default model

providers:
  deepseek:
    enabled: true  # Enable/disable providers
    api_key: "${DEEPSEEK_API_KEY}"
    models: ["deepseek-chat", "deepseek-coder"]
    
  # Add your own providers easily
  your_custom_provider:
    type: "openai_compatible"
    base_url: "https://your-api.com/v1"
    api_key: "${YOUR_API_KEY}"
```

## 🆘 Troubleshooting

### Environment Issues
```bash
# Recreate environment
conda env remove -n langchain-rag-learning
python setup_environment.py
```

### API Key Issues
```bash
# Check configuration
make check-config

# Verify .env file
cat .env
```

### Port Conflicts
```bash
# Change ports in .env
API_PORT=8001
STREAMLIT_PORT=8502
```

### Local Model Issues
```bash
# Check Ollama status
ollama list
ollama serve  # Start Ollama server
```

## 💡 Tips for Learning

1. **Start Small**: Begin with DeepSeek or local models
2. **Experiment**: Try different document types and questions
3. **Read Code**: Explore the source code to understand implementation
4. **Contribute**: Add new features or improve existing ones
5. **Share**: Use this as a portfolio project

## 🎉 Next Steps

- Explore the learning modules in the web UI
- Try uploading your own documents
- Experiment with different LLM providers
- Check out the API documentation
- Contribute to the project on GitHub

---

**Happy Learning! 🚀**

Need help? Check the [full documentation](README.md) or open an issue on GitHub.