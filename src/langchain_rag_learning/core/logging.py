"""Structured logging configuration for the RAG learning system."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            ]:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)


class RAGLogger:
    """Enhanced logger for RAG system with structured logging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration."""
        settings = get_settings()
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        if settings.DEBUG:
            # Simple format for development
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            # Structured format for production
            console_formatter = StructuredFormatter()
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if settings.LOG_FILE:
            log_path = Path(settings.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def log_operation(
        self,
        operation: str,
        status: str,
        duration: Optional[float] = None,
        **kwargs
    ):
        """Log operation with structured data."""
        log_data = {
            'operation': operation,
            'status': status,
            **kwargs
        }
        
        if duration is not None:
            log_data['duration_ms'] = round(duration * 1000, 2)
        
        if status == 'success':
            self.info(f"Operation {operation} completed successfully", **log_data)
        elif status == 'error':
            self.error(f"Operation {operation} failed", **log_data)
        else:
            self.info(f"Operation {operation} status: {status}", **log_data)
    
    def log_query(
        self,
        query_id: str,
        question: str,
        response_time: float,
        status: str = 'success',
        **kwargs
    ):
        """Log RAG query with structured data."""
        log_data = {
            'query_id': query_id,
            'question_length': len(question),
            'response_time_ms': round(response_time * 1000, 2),
            'status': status,
            **kwargs
        }
        
        self.log_operation('rag_query', status, response_time, **log_data)
    
    def log_document_processing(
        self,
        document_id: str,
        filename: str,
        status: str,
        processing_time: Optional[float] = None,
        **kwargs
    ):
        """Log document processing with structured data."""
        log_data = {
            'document_id': document_id,
            'filename': filename,
            'status': status,
            **kwargs
        }
        
        self.log_operation('document_processing', status, processing_time, **log_data)
    
    def log_embedding(
        self,
        text_length: int,
        embedding_time: float,
        model_name: str,
        status: str = 'success',
        **kwargs
    ):
        """Log embedding generation with structured data."""
        log_data = {
            'text_length': text_length,
            'embedding_time_ms': round(embedding_time * 1000, 2),
            'model_name': model_name,
            'status': status,
            **kwargs
        }
        
        self.log_operation('embedding_generation', status, embedding_time, **log_data)
    
    def log_retrieval(
        self,
        query: str,
        retrieval_time: float,
        results_count: int,
        retrieval_type: str = 'hybrid',
        **kwargs
    ):
        """Log retrieval operation with structured data."""
        log_data = {
            'query_length': len(query),
            'retrieval_time_ms': round(retrieval_time * 1000, 2),
            'results_count': results_count,
            'retrieval_type': retrieval_type,
            **kwargs
        }
        
        self.log_operation('retrieval', 'success', retrieval_time, **log_data)
    
    def log_llm_call(
        self,
        provider: str,
        model: str,
        prompt_length: int,
        response_length: int,
        call_time: float,
        **kwargs
    ):
        """Log LLM API call with structured data."""
        log_data = {
            'provider': provider,
            'model': model,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'call_time_ms': round(call_time * 1000, 2),
            **kwargs
        }
        
        self.log_operation('llm_call', 'success', call_time, **log_data)


class LoggerManager:
    """Manage multiple loggers for different components."""
    
    def __init__(self):
        self._loggers: Dict[str, RAGLogger] = {}
    
    def get_logger(self, name: str) -> RAGLogger:
        """Get or create logger for component."""
        if name not in self._loggers:
            self._loggers[name] = RAGLogger(name)
        return self._loggers[name]
    
    def configure_all_loggers(self):
        """Reconfigure all existing loggers."""
        for logger in self._loggers.values():
            logger._setup_logger()


# Global logger manager
logger_manager = LoggerManager()


def get_logger(name: str) -> RAGLogger:
    """Get logger for component."""
    return logger_manager.get_logger(name)


# Convenience loggers for common components
system_logger = get_logger('system')
api_logger = get_logger('api')
rag_logger = get_logger('rag')
document_logger = get_logger('document')
embedding_logger = get_logger('embedding')
retrieval_logger = get_logger('retrieval')
llm_logger = get_logger('llm')
learning_logger = get_logger('learning')


def setup_logging():
    """Setup logging configuration."""
    logger_manager.configure_all_loggers()
    system_logger.info("Logging system initialized")


def log_startup_info():
    """Log system startup information."""
    settings = get_settings()
    
    system_logger.info(
        "RAG Learning System starting up",
        app_name=getattr(settings, 'APP_NAME', 'RAG Learning System'),
        app_version=getattr(settings, 'APP_VERSION', '0.1.0'),
        debug_mode=settings.DEBUG,
        log_level=settings.LOG_LEVEL
    )


def log_shutdown_info():
    """Log system shutdown information."""
    system_logger.info("RAG Learning System shutting down")


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: RAGLogger, **context):
        self.logger = logger
        self.context = context
        self.original_logger = None
    
    def __enter__(self):
        # Store original logger methods
        self.original_methods = {}
        for method_name in ['debug', 'info', 'warning', 'error', 'critical']:
            self.original_methods[method_name] = getattr(self.logger, method_name)
            
            # Create wrapped method that includes context
            def create_wrapped_method(original_method):
                def wrapped_method(message: str, **kwargs):
                    merged_kwargs = {**self.context, **kwargs}
                    return original_method(message, **merged_kwargs)
                return wrapped_method
            
            setattr(self.logger, method_name, create_wrapped_method(self.original_methods[method_name]))
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods
        for method_name, original_method in self.original_methods.items():
            setattr(self.logger, method_name, original_method)