"""LLM response caching system."""

import hashlib
import json  # JSON parsing and serialization
import pickle
import time  # Time utilities for performance measurement
from typing import Any, Dict, Optional, Union  # Type hints for better code documentation
from datetime import datetime, timedelta  # Time utilities for performance measurement

try:
    import aioredis  # Regular expressions for text processing
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

from ..core.config import get_settings  # Regular expressions for text processing
from ..core.logging import get_logger  # Structured logging for debugging and monitoring
from ..core.models import LLMResponse  # Regular expressions for text processing

logger = get_logger(__name__)
settings = get_settings()


class LLMCache:
    """Cache for LLM responses to reduce API calls and improve performance."""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: Optional[int] = None):
        """
          Init   function implementation.
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.ttl = ttl or settings.llm_cache_ttl
        self.redis_client = None
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._use_redis = True
        
        # Try to initialize Redis
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            if not AIOREDIS_AVAILABLE:
                logger.warning("aioredis not available, using local cache only")
                self._use_redis = False
                return
            # This will be connected when first used
            self.redis_client = None
            logger.info("Redis cache configured")
        except Exception as e:
            logger.warning(f"Redis not available, using local cache: {e}")
            self._use_redis = False
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if not self._use_redis or not AIOREDIS_AVAILABLE:
            return None
        
        if self.redis_client is None:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False  # We'll handle encoding ourselves
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._use_redis = False
                return None
        
        return self.redis_client
    
    def generate_cache_key(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a cache key for the given parameters."""
        
        # Create a dictionary of all parameters that affect the response
        cache_params = {
            'prompt': prompt,
            'provider': provider,
            'model_name': model_name,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        # Add other relevant kwargs
        relevant_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']
        }
        cache_params.update(relevant_kwargs)
        
        # Create a stable string representation
        cache_string = json.dumps(cache_params, sort_keys=True, default=str)
        
        # Generate hash
        cache_key = hashlib.sha256(cache_string.encode()).hexdigest()
        
        return f"llm_cache:{cache_key}"
    
    async def get(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response."""
        try:
            # Try Redis first
            redis_client = await self._get_redis_client()
            if redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    # Deserialize the response
                    response_data = pickle.loads(cached_data)
                    logger.debug(f"Redis cache hit: {cache_key}")
                    return LLMResponse(**response_data)
            
            # Fallback to local cache
            if cache_key in self._local_cache:
                cache_entry = self._local_cache[cache_key]
                
                # Check if expired
                if cache_entry['expires_at'] > time.time():
                    logger.debug(f"Local cache hit: {cache_key}")
                    return LLMResponse(**cache_entry['data'])
                else:
                    # Remove expired entry
                    del self._local_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, cache_key: str, response: LLMResponse) -> bool:
        """Set cached response."""
        try:
            response_data = response.dict()
            
            # Try Redis first
            redis_client = await self._get_redis_client()
            if redis_client:
                serialized_data = pickle.dumps(response_data)
                await redis_client.setex(cache_key, self.ttl, serialized_data)
                logger.debug(f"Cached to Redis: {cache_key}")
                return True
            
            # Fallback to local cache
            self._local_cache[cache_key] = {
                'data': response_data,
                'expires_at': time.time() + self.ttl
            }
            
            # Clean up expired entries periodically
            await self._cleanup_local_cache()
            
            logger.debug(f"Cached locally: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, cache_key: str) -> bool:
        """Delete cached response."""
        try:
            deleted = False
            
            # Try Redis first
            redis_client = await self._get_redis_client()
            if redis_client:
                result = await redis_client.delete(cache_key)
                deleted = result > 0
            
            # Also remove from local cache
            if cache_key in self._local_cache:
                del self._local_cache[cache_key]
                deleted = True
            
            if deleted:
                logger.debug(f"Deleted from cache: {cache_key}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        try:
            cleared_count = 0
            
            # Clear Redis
            redis_client = await self._get_redis_client()
            if redis_client:
                if pattern:
                    keys = await redis_client.keys(pattern)
                    if keys:
                        cleared_count += await redis_client.delete(*keys)
                else:
                    # Clear all LLM cache keys
                    keys = await redis_client.keys("llm_cache:*")
                    if keys:
                        cleared_count += await redis_client.delete(*keys)
            
            # Clear local cache
            if pattern:
                # Simple pattern matching for local cache
                keys_to_delete = [
                    key for key in self._local_cache.keys()
                    if pattern.replace('*', '') in key
                ]
                for key in keys_to_delete:
                    del self._local_cache[key]
                    cleared_count += 1
            else:
                cleared_count += len(self._local_cache)
                self._local_cache.clear()
            
            logger.info(f"Cleared {cleared_count} cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'local_cache_size': len(self._local_cache),
            'redis_available': self._use_redis,
            'ttl': self.ttl
        }
        
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                # Get Redis info
                info = await redis_client.info('memory')
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_keys': await redis_client.dbsize()
                })
                
                # Count LLM cache keys
                llm_keys = await redis_client.keys("llm_cache:*")
                stats['redis_llm_cache_keys'] = len(llm_keys)
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            stats['redis_error'] = str(e)
        
        return stats
    
    async def _cleanup_local_cache(self):
        """Clean up expired entries from local cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._local_cache.items()
            if entry['expires_at'] <= current_time
        ]
        
        for key in expired_keys:
            del self._local_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired local cache entries")
    
    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Closed Redis connection")


class CacheManager:
    """Manager for different cache strategies."""
    
    def __init__(self):
        """
          Init   function implementation.
        """
        self.caches: Dict[str, LLMCache] = {}
        self.default_cache = LLMCache()
    
    def get_cache(self, cache_name: str = "default") -> LLMCache:
        """Get or create a named cache."""
        if cache_name == "default":
            return self.default_cache
        
        if cache_name not in self.caches:
            self.caches[cache_name] = LLMCache()
        
        return self.caches[cache_name]
    
    async def clear_all_caches(self) -> int:
        """Clear all caches."""
        total_cleared = 0
        
        # Clear default cache
        total_cleared += await self.default_cache.clear()
        
        # Clear named caches
        for cache in self.caches.values():
            total_cleared += await cache.clear()
        
        return total_cleared
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {
            'default': await self.default_cache.get_stats()
        }
        
        for name, cache in self.caches.items():
            stats[name] = await cache.get_stats()
        
        return stats
    
    async def close_all(self):
        """Close all cache connections."""
        await self.default_cache.close()
        
        for cache in self.caches.values():
            await cache.close()


# Global cache manager
cache_manager = CacheManager()


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(self, cache: Optional[LLMCache] = None, ttl: Optional[int] = None):
        """
          Init   function implementation.
        """
        self.cache = cache or cache_manager.get_cache()
        self.ttl = ttl
    
    def __call__(self, func):
        """
          Call   function implementation.
        """
        async def wrapper(*args, **kwargs):
            """
            Async wrapper function implementation.
            """
            # Generate cache key from function name and arguments
            cache_key_data = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            cache_key = f"func_cache:{cache_key}"
            
            # Try to get from cache
            try:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            try:
                await self.cache.set(cache_key, result)
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
            
            return result
        
        return wrapper


def cached(cache: Optional[LLMCache] = None, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    return CacheDecorator(cache, ttl)