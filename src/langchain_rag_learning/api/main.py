"""
FastAPI main application for the LangChain RAG Learning Platform.

This module implements a production-ready REST API using FastAPI framework:
- Async request handling for high performance
- Automatic API documentation generation
- Request/response validation with Pydantic
- Error handling and HTTP status codes
- CORS middleware for web browser compatibility
- Dependency injection for clean architecture

API Endpoints:
- GET /: Root endpoint with basic information
- GET /health: Health check for monitoring
- GET /config: Current configuration status
- POST /query: Main RAG query endpoint
- GET /providers: List available LLM providers

Technical Features:
- ASGI (Asynchronous Server Gateway Interface)
- OpenAPI/Swagger documentation
- Pydantic models for data validation
- Structured error responses
- Graceful fallback mechanisms
"""

from fastapi import FastAPI, HTTPException  # FastAPI framework and HTTP exceptions  # FastAPI framework for REST API
from fastapi.middleware.cors import CORSMiddleware  # Cross-Origin Resource Sharing  # FastAPI framework for REST API
from pydantic import BaseModel  # Data validation and serialization  # Data validation and serialization
from typing import Dict, Any, List  # Type hints for better code documentation  # Type hints for better code documentation
import sys  # System-specific parameters and functions  # System-specific parameters and functions
import os   # Operating system interface  # Operating system interface

# Add src directory to Python path for module imports
# This allows importing from the project's source directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import project configuration modules with fallback handling
try:
    from langchain_rag_learning.core.config import get_settings, get_llm_config  # LangChain framework for LLM applications
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback mock functions for basic functionality when imports fail
    def get_settings():
        """Mock settings class for fallback functionality."""
        class MockSettings:
            """
            MockSettings class implementation.
            """
            API_HOST = "0.0.0.0"  # Default host for all interfaces
            API_PORT = 8000       # Default HTTP port
        return MockSettings()
    
    def get_llm_config():
        """Mock LLM configuration for fallback functionality."""
        class MockLLMConfig:
            """
            MockLLMConfig class implementation.
            """
            def get_enabled_providers(self):
                """Return default enabled providers."""
                return ["deepseek", "ollama"]  # Cost-effective options
            def get_default_provider(self):
                """Return default provider."""
                return "deepseek"  # Most affordable for learning
        return MockLLMConfig()

# Initialize FastAPI application with metadata for documentation
app = FastAPI(
    title="LangChain RAG Learning API",  # API title shown in documentation
    description="A comprehensive learning platform for LangChain and RAG technologies",
    version="0.1.0"  # Semantic versioning
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# CORS allows web browsers to make requests from different domains
# This is essential for web applications that need to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow requests from any origin (use specific domains in production)
    allow_credentials=True,     # Allow cookies and authentication headers
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],        # Allow all request headers
)

# Pydantic models for request/response validation and documentation
# These models automatically validate incoming data and generate API documentation

class HealthResponse(BaseModel):
    """
    Response model for health check endpoints.
    
    Pydantic automatically validates the response data and generates
    OpenAPI schema documentation for the API.
    """
    status: str   # Health status (e.g., "healthy", "unhealthy")
    message: str  # Descriptive message about the system state
    version: str

class ConfigResponse(BaseModel):
    """
    ConfigResponse class implementation.
    """
    enabled_providers: List[str]
    default_provider: str

class QueryRequest(BaseModel):
    """
    QueryRequest class implementation.
    """
    question: str
    provider: str = "deepseek"

class QueryResponse(BaseModel):
    """
    QueryResponse class implementation.
    """
    answer: str
    provider: str
    sources: List[str] = []

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="ok",
        message="LangChain RAG Learning API is running!",
        version="0.1.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running normally",
        version="0.1.0"
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    try:
        llm_config = get_llm_config()
        return ConfigResponse(
            enabled_providers=llm_config.get_enabled_providers(),
            default_provider=llm_config.get_default_provider()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    # This is a placeholder implementation
    # In a real implementation, this would use the RAG engine
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Mock response for now
    mock_answer = f"This is a mock response to your question: '{request.question}'. " \
                  f"The system would use the {request.provider} provider to generate a real answer."
    
    return QueryResponse(
        answer=mock_answer,
        provider=request.provider,
        sources=["mock_document_1.pdf", "mock_document_2.md"]
    )

@app.get("/providers")
async def list_providers():
    """List available LLM providers."""
    try:
        llm_config = get_llm_config()
        providers = llm_config.get_all_providers()
        
        provider_info = {}
        for name, config in providers.items():
            provider_info[name] = {
                "name": config.get("name", name),
                "type": config.get("type", "unknown"),
                "enabled": config.get("enabled", False),
                "models": config.get("models", [])
            }
        
        return provider_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing providers: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )