"""Custom exceptions for the LangChain RAG learning system."""

from typing import Any, Dict, Optional


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ConfigurationError(RAGException):
    """Configuration related errors."""
    pass


class ValidationError(RAGException):
    """Data validation errors."""
    pass


class DocumentProcessingError(RAGException):
    """Document processing related errors."""
    pass


class FileUploadError(DocumentProcessingError):
    """File upload specific errors."""
    pass


class FileFormatError(DocumentProcessingError):
    """File format specific errors."""
    pass


class ChunkingError(DocumentProcessingError):
    """Document chunking errors."""
    pass


class EmbeddingError(RAGException):
    """Embedding generation errors."""
    pass


class VectorStoreError(RAGException):
    """Vector store related errors."""
    pass


class IndexError(VectorStoreError):
    """Vector index errors."""
    pass


class SearchError(VectorStoreError):
    """Vector search errors."""
    pass


class RetrievalError(RAGException):
    """Retrieval related errors."""
    pass


class DenseRetrievalError(RetrievalError):
    """Dense retrieval errors."""
    pass


class SparseRetrievalError(RetrievalError):
    """Sparse retrieval errors."""
    pass


class RerankingError(RetrievalError):
    """Reranking errors."""
    pass


class LLMError(RAGException):
    """LLM related errors."""
    pass


class LLMProviderError(LLMError):
    """LLM provider specific errors."""
    pass


class LLMQuotaExceededError(LLMError):
    """LLM quota exceeded errors."""
    pass


class LLMTimeoutError(LLMError):
    """LLM timeout errors."""
    pass


class GenerationError(LLMError):
    """Text generation errors."""
    pass


class DatabaseError(RAGException):
    """Database related errors."""
    pass


class ConnectionError(DatabaseError):
    """Database connection errors."""
    pass


class QueryError(DatabaseError):
    """Database query errors."""
    pass


class AuthenticationError(RAGException):
    """Authentication related errors."""
    pass


class AuthorizationError(RAGException):
    """Authorization related errors."""
    pass


class APIError(RAGException):
    """API related errors."""
    pass


class RateLimitError(APIError):
    """Rate limiting errors."""
    pass


class LearningModuleError(RAGException):
    """Learning module related errors."""
    pass


class ContentNotFoundError(LearningModuleError):
    """Content not found errors."""
    pass


class ProgressTrackingError(LearningModuleError):
    """Progress tracking errors."""
    pass


class EvaluationError(RAGException):
    """Evaluation related errors."""
    pass


class MetricsError(EvaluationError):
    """Metrics calculation errors."""
    pass


class BenchmarkError(EvaluationError):
    """Benchmark execution errors."""
    pass


class SystemError(RAGException):
    """System level errors."""
    pass


class ResourceError(SystemError):
    """Resource related errors."""
    pass


class MonitoringError(SystemError):
    """Monitoring related errors."""
    pass


# Error code mappings for common errors
ERROR_CODES = {
    # Configuration errors
    "CONFIG_MISSING": "Required configuration is missing",
    "CONFIG_INVALID": "Configuration value is invalid",
    "API_KEY_MISSING": "API key is missing or invalid",
    
    # Document processing errors
    "FILE_TOO_LARGE": "File size exceeds maximum allowed size",
    "FILE_FORMAT_UNSUPPORTED": "File format is not supported",
    "FILE_CORRUPTED": "File appears to be corrupted",
    "EXTRACTION_FAILED": "Text extraction from file failed",
    
    # Vector store errors
    "INDEX_NOT_FOUND": "Vector index not found",
    "INDEX_CORRUPTED": "Vector index is corrupted",
    "SEARCH_FAILED": "Vector search operation failed",
    
    # LLM errors
    "LLM_UNAVAILABLE": "LLM service is unavailable",
    "LLM_QUOTA_EXCEEDED": "LLM API quota exceeded",
    "LLM_TIMEOUT": "LLM request timed out",
    "GENERATION_FAILED": "Text generation failed",
    
    # Database errors
    "DB_CONNECTION_FAILED": "Database connection failed",
    "DB_QUERY_FAILED": "Database query failed",
    "DB_CONSTRAINT_VIOLATION": "Database constraint violation",
    
    # Authentication/Authorization errors
    "AUTH_TOKEN_INVALID": "Authentication token is invalid",
    "AUTH_TOKEN_EXPIRED": "Authentication token has expired",
    "PERMISSION_DENIED": "Permission denied for this operation",
    
    # API errors
    "RATE_LIMIT_EXCEEDED": "API rate limit exceeded",
    "REQUEST_INVALID": "Request format is invalid",
    "RESOURCE_NOT_FOUND": "Requested resource not found",
    
    # System errors
    "RESOURCE_EXHAUSTED": "System resources exhausted",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "INTERNAL_ERROR": "Internal system error occurred"
}


def get_error_message(error_code: str) -> str:
    """Get human-readable error message for error code."""
    return ERROR_CODES.get(error_code, "Unknown error occurred")


def create_error_response(
    exception: RAGException,
    include_details: bool = False
) -> Dict[str, Any]:
    """Create standardized error response."""
    response = {
        "success": False,
        "error": {
            "type": exception.__class__.__name__,
            "message": exception.message,
            "code": exception.error_code
        }
    }
    
    if include_details and exception.details:
        response["error"]["details"] = exception.details
    
    return response