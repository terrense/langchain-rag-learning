"""FastAPI main application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from langchain_rag_learning.core.config import get_settings, get_llm_config
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for basic functionality
    def get_settings():
        class MockSettings:
            API_HOST = "0.0.0.0"
            API_PORT = 8000
        return MockSettings()
    
    def get_llm_config():
        class MockLLMConfig:
            def get_enabled_providers(self):
                return ["deepseek", "ollama"]
            def get_default_provider(self):
                return "deepseek"
        return MockLLMConfig()

app = FastAPI(
    title="LangChain RAG Learning API",
    description="A comprehensive learning platform for LangChain and RAG technologies",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class ConfigResponse(BaseModel):
    enabled_providers: List[str]
    default_provider: str

class QueryRequest(BaseModel):
    question: str
    provider: str = "deepseek"

class QueryResponse(BaseModel):
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