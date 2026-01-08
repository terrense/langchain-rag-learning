# API Technical Guide: FastAPI Implementation

## Overview
This document explains the technical implementation of the FastAPI-based REST API, covering advanced Python concepts, web development patterns, and API design principles.

## FastAPI Framework Concepts

### 1. ASGI (Asynchronous Server Gateway Interface)
```python
from fastapi import FastAPI

app = FastAPI(
    title="LangChain RAG Learning API",
    description="A comprehensive learning platform for LangChain and RAG technologies",
    version="0.1.0"
)
```
- **ASGI**: Modern Python web server interface for async applications
- **Async/Await**: Non-blocking I/O operations for better performance
- **Metadata**: API documentation generation from code annotations
- **OpenAPI**: Automatic API documentation and validation

### 2. Dependency Injection System
```python
from fastapi import Depends, HTTPException

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Dependency injection - FastAPI automatically provides token
    return verify_token(token)

@app.get("/protected")
async def protected_route(user = Depends(get_current_user)):
    # user is automatically injected by FastAPI
    return {"user": user}
```
- **Dependency Injection**: Automatic parameter resolution
- **Inversion of Control**: Framework manages object creation
- **Reusability**: Dependencies can be shared across endpoints
- **Testing**: Easy to mock dependencies for testing

### 3. Pydantic Models for Request/Response
```python
from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    provider: str = "deepseek"  # Default value
    max_tokens: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    provider: str
    sources: List[str] = []  # Default empty list
    metadata: Dict[str, Any] = {}
```
- **Data Validation**: Automatic request/response validation
- **Serialization**: JSON ↔ Python object conversion
- **Documentation**: Auto-generated API docs from models
- **Type Safety**: Compile-time type checking

### 4. Async/Await Pattern
```python
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # Async function can handle concurrent requests
    try:
        # Await suspends execution without blocking thread
        result = await llm_service.generate_response(request.question)
        return QueryResponse(
            answer=result.content,
            provider=request.provider,
            sources=result.sources
        )
    except Exception as e:
        # Proper error handling with HTTP status codes
        raise HTTPException(status_code=500, detail=str(e))
```
- **Concurrency**: Handle multiple requests simultaneously
- **Non-blocking**: Doesn't block thread while waiting for I/O
- **Event Loop**: Single-threaded concurrent execution
- **Performance**: Better resource utilization than threading

## HTTP Protocol and REST API Design

### 1. HTTP Methods and Semantics
```python
@app.get("/health")          # GET: Retrieve data (idempotent)
async def health_check():
    return {"status": "healthy"}

@app.post("/query")          # POST: Create/process data
async def query_rag(request: QueryRequest):
    # Process the query and return result
    pass

@app.put("/config/{key}")    # PUT: Update/replace resource
async def update_config(key: str, value: str):
    # Update configuration value
    pass

@app.delete("/cache")        # DELETE: Remove resource
async def clear_cache():
    # Clear application cache
    pass
```
- **HTTP Verbs**: Semantic meaning for different operations
- **Idempotency**: GET/PUT operations can be repeated safely
- **Status Codes**: Standardized response codes (200, 404, 500, etc.)
- **RESTful Design**: Resource-based URL structure

### 2. Middleware and Request Processing
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # CORS policy
    allow_credentials=True,        # Cookie/auth headers
    allow_methods=["*"],           # HTTP methods
    allow_headers=["*"],           # Request headers
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Middleware runs before and after request processing
    start_time = time.time()
    response = await call_next(request)  # Process request
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```
- **Middleware Stack**: Request/response processing pipeline
- **CORS**: Cross-Origin Resource Sharing for web browsers
- **Request Lifecycle**: Pre/post processing hooks
- **Headers**: HTTP metadata for requests and responses

### 3. Error Handling and HTTP Status Codes
```python
from fastapi import HTTPException, status

@app.post("/query")
async def query_rag(request: QueryRequest):
    if not request.question.strip():
        # 400 Bad Request: Client error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        result = await process_query(request)
        return result
    except ServiceUnavailableError:
        # 503 Service Unavailable: Temporary server error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is temporarily unavailable"
        )
    except Exception as e:
        # 500 Internal Server Error: Unexpected server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
```
- **HTTP Status Codes**: Standardized error communication
- **Exception Handling**: Convert Python exceptions to HTTP responses
- **Client vs Server Errors**: 4xx (client) vs 5xx (server) errors
- **Error Details**: Descriptive error messages for debugging

## Advanced Python Concepts in API

### 1. Context Managers for Resource Management
```python
import aiohttp
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_http_client():
    """Async context manager for HTTP client lifecycle."""
    async with aiohttp.ClientSession() as session:
        try:
            yield session  # Provide session to caller
        finally:
            # Cleanup happens automatically
            pass

async def call_external_api():
    async with get_http_client() as client:
        async with client.get("https://api.example.com") as response:
            return await response.json()
```
- **Resource Management**: Automatic cleanup of connections
- **Async Context Managers**: __aenter__ and __aexit__ methods
- **Connection Pooling**: Reuse HTTP connections for efficiency
- **Exception Safety**: Cleanup occurs even if errors happen

### 2. Generators and Streaming Responses
```python
from fastapi.responses import StreamingResponse
import json

async def generate_streaming_response(query: str):
    """Async generator for streaming LLM responses."""
    async for chunk in llm_service.stream_response(query):
        # Yield each chunk as it becomes available
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

@app.post("/stream")
async def stream_query(request: QueryRequest):
    return StreamingResponse(
        generate_streaming_response(request.question),
        media_type="text/plain"
    )
```
- **Async Generators**: yield in async functions
- **Streaming**: Send data as it becomes available
- **Memory Efficiency**: Don't buffer entire response
- **Real-time**: Lower latency for long responses

### 3. Background Tasks
```python
from fastapi import BackgroundTasks

async def log_query(query: str, response: str):
    """Background task for logging (doesn't block response)."""
    # This runs after the response is sent to client
    await database.log_interaction(query, response)

@app.post("/query")
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    result = await process_query(request)
    
    # Schedule background task (non-blocking)
    background_tasks.add_task(log_query, request.question, result.answer)
    
    return result  # Response sent immediately
```
- **Background Processing**: Non-blocking task execution
- **Response Time**: Faster API responses
- **Task Queue**: Built-in task scheduling
- **Fire and Forget**: Tasks run after response is sent

## Security Implementation

### 1. Authentication and Authorization
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user information."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
async def protected_endpoint(current_user: str = Depends(verify_token)):
    return {"user": current_user, "message": "Access granted"}
```
- **JWT Tokens**: Stateless authentication
- **Bearer Authentication**: Standard HTTP auth scheme
- **Token Validation**: Cryptographic signature verification
- **Dependency Injection**: Automatic token verification

### 2. Input Validation and Sanitization
```python
from pydantic import validator, Field
import re

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    provider: str = Field(default="deepseek", regex="^[a-zA-Z0-9_-]+$")
    
    @validator('question')
    def validate_question(cls, v):
        # Remove potentially dangerous characters
        if re.search(r'[<>"\']', v):
            raise ValueError("Question contains invalid characters")
        return v.strip()
    
    @validator('provider')
    def validate_provider(cls, v):
        allowed_providers = ["openai", "anthropic", "deepseek", "ollama"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v
```
- **Input Validation**: Prevent malicious input
- **Regular Expressions**: Pattern matching for validation
- **Whitelist Approach**: Only allow known-good values
- **XSS Prevention**: Remove dangerous characters

### 3. Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def query_rag(request: Request, query_request: QueryRequest):
    # Rate limiting applied automatically
    return await process_query(query_request)
```
- **Rate Limiting**: Prevent API abuse
- **Per-IP Limits**: Track usage by client IP
- **Sliding Window**: Time-based request counting
- **Automatic Enforcement**: Middleware handles rate limiting

## Performance Optimization

### 1. Connection Pooling
```python
import aiohttp
from aiohttp import TCPConnector

# Global HTTP client with connection pooling
connector = TCPConnector(
    limit=100,              # Total connection pool size
    limit_per_host=30,      # Connections per host
    ttl_dns_cache=300,      # DNS cache TTL
    use_dns_cache=True,     # Enable DNS caching
)

http_client = aiohttp.ClientSession(connector=connector)

async def call_llm_api(prompt: str):
    async with http_client.post(
        "https://api.deepseek.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]}
    ) as response:
        return await response.json()
```
- **Connection Reuse**: Avoid connection overhead
- **Pool Management**: Limit concurrent connections
- **DNS Caching**: Reduce DNS lookup time
- **Keep-Alive**: Maintain persistent connections

### 2. Caching Strategies
```python
from functools import lru_cache
import asyncio
from typing import Dict, Any

# In-memory cache with TTL
cache: Dict[str, Dict[str, Any]] = {}

async def cached_llm_call(prompt: str, ttl: int = 3600) -> str:
    """Cache LLM responses to avoid repeated API calls."""
    cache_key = f"llm:{hash(prompt)}"
    
    # Check cache first
    if cache_key in cache:
        cached_data = cache[cache_key]
        if time.time() - cached_data["timestamp"] < ttl:
            return cached_data["response"]
        else:
            # Cache expired, remove entry
            del cache[cache_key]
    
    # Make API call and cache result
    response = await call_llm_api(prompt)
    cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    
    return response
```
- **Response Caching**: Avoid repeated expensive operations
- **TTL (Time To Live)**: Automatic cache expiration
- **Cache Keys**: Unique identifiers for cached data
- **Memory Management**: Remove expired entries

### 3. Async Database Operations
```python
import asyncpg
from contextlib import asynccontextmanager

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def initialize(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,      # Minimum connections
            max_size=20,     # Maximum connections
            command_timeout=60
        )
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        async with self.pool.acquire() as connection:
            yield connection
    
    async def log_query(self, query: str, response: str):
        """Log query to database asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO query_logs (query, response, timestamp) VALUES ($1, $2, $3)",
                query, response, datetime.utcnow()
            )
```
- **Connection Pooling**: Reuse database connections
- **Async I/O**: Non-blocking database operations
- **Resource Management**: Automatic connection cleanup
- **Scalability**: Handle many concurrent database operations

This technical guide covers the advanced concepts and patterns used in the FastAPI implementation, providing deep insights into modern Python web development practices.