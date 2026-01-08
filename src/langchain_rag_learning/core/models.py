"""Core data models for the LangChain RAG learning system."""

from datetime import datetime  # Time utilities for performance measurement
from enum import Enum
from typing import Any, Dict, List, Optional, Union  # Type hints for better code documentation
from uuid import UUID, uuid4

try:
    from pydantic import BaseModel, Field, validator  # Data validation and serialization
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    
    # Fallback implementations
    class BaseModel:
        """
        BaseModel class implementation.
        """
        def __init__(self, **kwargs):
            """
              Init   function implementation.
            """
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            """
            Dict function implementation.
            """
            return self.__dict__
    
    def Field(default=None, **kwargs):
        """
        Field function implementation.
        """
        return default
    
    def validator(field_name, **kwargs):
        """
        Validator function implementation.
        """
        def decorator(func):
            """
            Decorator function implementation.
            """
            return func
        return decorator


class UserLevel(str, Enum):
    """User learning levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(str, Enum):
    """Query types."""
    QUESTION = "question"
    SEARCH = "search"
    SUMMARY = "summary"
    EXPLANATION = "explanation"


class LLMProvider(str, Enum):
    """LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    DEEPSEEK = "deepseek"


class BaseEntity(BaseModel):
    """Base entity with common fields."""
    
    def __init__(self, **kwargs):
        """
          Init   function implementation.
        """
        if PYDANTIC_AVAILABLE:
            super().__init__(**kwargs)
        else:
            self.id = kwargs.get('id', str(uuid4()))
            self.created_at = kwargs.get('created_at', datetime.utcnow())
            self.updated_at = kwargs.get('updated_at', datetime.utcnow())
    
    if PYDANTIC_AVAILABLE:
        id: str = Field(default_factory=lambda: str(uuid4()))
        created_at: datetime = Field(default_factory=datetime.utcnow)
        updated_at: datetime = Field(default_factory=datetime.utcnow)
        
        class Config:
            """Pydantic configuration."""
            use_enum_values = True
            json_encoders = {
                datetime: lambda v: v.isoformat(),
                UUID: lambda v: str(v)
            }


class User(BaseEntity):
    """User model for the learning system."""
    
    def __init__(self, username: str, email: str, full_name: Optional[str] = None, 
        """
          Init   function implementation.
        """
                 learning_level: UserLevel = UserLevel.BEGINNER, **kwargs):
        if PYDANTIC_AVAILABLE:
            super().__init__(
                username=username,
                email=email,
                full_name=full_name,
                learning_level=learning_level,
                **kwargs
            )
        else:
            super().__init__(**kwargs)
            self.username = username
            self.email = email
            self.full_name = full_name
            self.learning_level = learning_level
            self.progress = kwargs.get('progress', {})
            self.preferences = kwargs.get('preferences', {})
            self.is_active = kwargs.get('is_active', True)
            self.last_active = kwargs.get('last_active')
    
    if PYDANTIC_AVAILABLE:
        username: str = Field(..., min_length=3, max_length=50)
        email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
        full_name: Optional[str] = Field(None, max_length=100)
        learning_level: UserLevel = Field(default=UserLevel.BEGINNER)
        progress: Dict[str, float] = Field(default_factory=dict)
        preferences: Dict[str, Any] = Field(default_factory=dict)
        is_active: bool = Field(default=True)
        last_active: Optional[datetime] = None
    
    @validator('progress')
    def validate_progress(cls, v):
        """Validate progress values are between 0 and 1."""
        if not PYDANTIC_AVAILABLE:
            return v
        for key, value in v.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Progress value for {key} must be between 0 and 1")
        return v


class KnowledgeBase(BaseEntity):
    """Knowledge base model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    owner_id: str = Field(...)
    document_count: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    embedding_model: str = Field(...)
    is_public: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseEntity):
    """Document model."""
    filename: str = Field(..., min_length=1, max_length=255)
    original_filename: str = Field(..., min_length=1, max_length=255)
    file_path: str = Field(...)
    file_size: int = Field(..., gt=0)
    file_type: str = Field(...)
    mime_type: str = Field(...)
    knowledge_base_id: str = Field(...)
    uploaded_by: str = Field(...)
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    processing_error: Optional[str] = None
    chunk_count: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('file_type')
    def validate_file_type(cls, v):
        """Validate file type is supported."""
        if not PYDANTIC_AVAILABLE:
            return v
        allowed_types = ['pdf', 'docx', 'txt', 'md', 'html']
        if v.lower() not in allowed_types:
            raise ValueError(f"File type {v} not supported. Allowed: {allowed_types}")
        return v.lower()


class DocumentChunk(BaseEntity):
    """Document chunk model for vector storage."""
    content: str = Field(..., min_length=1)
    document_id: str = Field(...)
    chunk_index: int = Field(..., ge=0)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('end_char')
    def validate_char_positions(cls, v, values):
        """Validate end_char is greater than start_char."""
        if not PYDANTIC_AVAILABLE:
            return v
        if 'start_char' in values and v <= values['start_char']:
            raise ValueError("end_char must be greater than start_char")
        return v


class QuerySession(BaseEntity):
    """Query session model for conversation tracking."""
    user_id: str = Field(...)
    knowledge_base_id: str = Field(...)
    session_name: Optional[str] = Field(None, max_length=100)
    query_count: int = Field(default=0, ge=0)
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Query(BaseEntity):
    """Query model."""
    session_id: str = Field(...)
    question: str = Field(..., min_length=1, max_length=2000)
    query_type: QueryType = Field(default=QueryType.QUESTION)
    answer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    retrieved_chunks: List[str] = Field(default_factory=list)
    llm_provider: Optional[LLMProvider] = None
    model_name: Optional[str] = None
    response_time: Optional[float] = Field(None, ge=0)
    retrieval_time: Optional[float] = Field(None, ge=0)
    generation_time: Optional[float] = Field(None, ge=0)
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    feedback_score: Optional[int] = Field(None, ge=1, le=5)
    feedback_comment: Optional[str] = Field(None, max_length=500)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningPath(BaseEntity):
    """Learning path model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    target_level: UserLevel = Field(...)
    prerequisites: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    estimated_hours: Optional[int] = Field(None, gt=0)
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TheoryContent(BaseEntity):
    """Theory content model."""
    title: str = Field(..., min_length=1, max_length=200)
    topic: str = Field(..., min_length=1, max_length=100)
    level: UserLevel = Field(...)
    content: str = Field(..., min_length=1)
    content_type: str = Field(default="markdown")
    examples: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    next_topics: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    estimated_read_time: Optional[int] = Field(None, gt=0)
    tags: List[str] = Field(default_factory=list)
    is_published: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tutorial(BaseEntity):
    """Tutorial model."""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    level: UserLevel = Field(...)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    code_examples: List[Dict[str, str]] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    estimated_time: Optional[int] = Field(None, gt=0)
    tags: List[str] = Field(default_factory=list)
    is_published: bool = Field(default=False)
    completion_criteria: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserProgress(BaseEntity):
    """User progress tracking model."""
    user_id: str = Field(...)
    content_id: str = Field(...)
    content_type: str = Field(...)  # 'theory', 'tutorial', 'learning_path'
    progress_percentage: float = Field(default=0.0, ge=0, le=1)
    completed_steps: List[str] = Field(default_factory=list)
    time_spent: int = Field(default=0, ge=0)  # in seconds
    last_accessed: Optional[datetime] = None
    is_completed: bool = Field(default=False)
    completion_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseEntity):
    """System metrics model."""
    metric_name: str = Field(..., min_length=1, max_length=100)
    metric_value: Union[int, float, str] = Field(...)
    metric_type: str = Field(...)  # 'counter', 'gauge', 'histogram'
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Response models for API
class RAGResponse(BaseModel):
    """RAG query response model."""
    query_id: str
    answer: str
    sources: List[DocumentChunk]
    confidence: Optional[float] = None
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Document ingestion result model."""
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Evaluation result model."""
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    test_name: str
    metrics: Dict[str, float]
    test_cases_count: int
    passed_count: int
    failed_count: int
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """LLM response model."""
    
    def __init__(self, content: str, model_name: str, provider: LLMProvider, 
        """
          Init   function implementation.
        """
                 response_time: float, usage: Dict[str, int] = None, 
                 metadata: Dict[str, Any] = None, **kwargs):
        if PYDANTIC_AVAILABLE:
            super().__init__(
                content=content,
                model_name=model_name,
                provider=provider,
                response_time=response_time,
                usage=usage or {},
                metadata=metadata or {},
                **kwargs
            )
        else:
            self.content = content
            self.model_name = model_name
            self.provider = provider
            self.response_time = response_time
            self.usage = usage or {}
            self.metadata = metadata or {}
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    if PYDANTIC_AVAILABLE:
        content: str
        model_name: str
        provider: LLMProvider
        usage: Dict[str, int] = Field(default_factory=dict)
        response_time: float
        metadata: Dict[str, Any] = Field(default_factory=dict)