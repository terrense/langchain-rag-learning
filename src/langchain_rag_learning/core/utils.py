"""Utility functions and classes for the RAG learning system."""

import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

from .exceptions import RAGException, SystemError

# Type variable for generic functions
T = TypeVar('T')


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


def generate_hash(content: str) -> str:
    """Generate SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters."""
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')
    # Ensure it's not empty
    if not safe_name:
        safe_name = f"file_{generate_id()[:8]}"
    return safe_name


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def __str__(self):
        if self.duration is not None:
            return f"{self.name}: {format_duration(self.duration)}"
        return f"{self.name}: Not completed"


class AsyncTimer:
    """Async context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def __str__(self):
        if self.duration is not None:
            return f"{self.name}: {format_duration(self.duration)}"
        return f"{self.name}: Not completed"


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator for retrying functions with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        break
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator for retrying async functions with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        break
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise SystemError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens."""
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.rate


class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: float = 3600.0):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires']:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        self._cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def cleanup(self) -> int:
        """Remove expired entries and return count."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry['expires']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current = min(self.total, self.current + increment)
        self.last_update = time.time()
    
    def set_progress(self, current: int) -> None:
        """Set absolute progress."""
        self.current = min(self.total, max(0, current))
        self.last_update = time.time()
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_total_time(self) -> Optional[float]:
        """Estimate total time based on current progress."""
        if self.current == 0:
            return None
        
        elapsed = self.elapsed_time
        rate = self.current / elapsed
        return self.total / rate
    
    @property
    def eta(self) -> Optional[float]:
        """Estimate time remaining."""
        total_time = self.estimated_total_time
        if total_time is None:
            return None
        
        return max(0, total_time - self.elapsed_time)
    
    def __str__(self) -> str:
        """String representation of progress."""
        percentage = self.percentage
        elapsed = format_duration(self.elapsed_time)
        
        if self.eta is not None:
            eta_str = format_duration(self.eta)
            return f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - {elapsed} elapsed, {eta_str} remaining"
        else:
            return f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - {elapsed} elapsed"


def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file type based on extension."""
    if not filename:
        return False
    
    extension = Path(filename).suffix.lower().lstrip('.')
    return extension in [ext.lower() for ext in allowed_extensions]


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input text."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes and control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    path = Path(filename)
    
    return {
        'filename': path.name,
        'stem': path.stem,
        'suffix': path.suffix.lower(),
        'size_hint': None,  # Will be filled when file is processed
        'extracted_at': datetime.utcnow().isoformat()
    }


class BatchProcessor:
    """Process items in batches with progress tracking."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    async def process_async(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items asynchronously in batches."""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch
            batch_results = []
            for item in batch:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = processor(item)
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append(e)
            
            results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress_callback(len(results), total_items)
        
        return results