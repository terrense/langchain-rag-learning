"""LLM Manager for handling multiple providers and load balancing."""

import asyncio  # Async programming support for concurrent operations
import time  # Time utilities for performance measurement
from typing import Any, Dict, List, Optional, Union  # Type hints for better code documentation
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta  # Time utilities for performance measurement

from ..core.config import get_settings  # Regular expressions for text processing
from ..core.exceptions import LLMError, ConfigurationError  # Regular expressions for text processing
from ..core.logging import get_logger  # Structured logging for debugging and monitoring
from ..core.models import LLMProvider, LLMResponse  # Regular expressions for text processing
from .providers import BaseLLMProvider, create_provider  # Regular expressions for text processing
from .cache import LLMCache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    consecutive_failures: int = 0
    is_healthy: bool = True
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests


@dataclass
class RateLimiter:
    """Rate limiter for API calls."""
    max_requests: int
    time_window: int  # seconds
    requests: deque = field(default_factory=deque)
    
    def can_make_request(self) -> bool:
        """Check if a request can be made."""
        now = time.time()
        
        # Remove old requests outside the time window
        while self.requests and self.requests[0] <= now - self.time_window:
            self.requests.popleft()
        
        return len(self.requests) < self.max_requests
    
    def record_request(self) -> None:
        """Record a new request."""
        self.requests.append(time.time())
    
    def time_until_next_request(self) -> float:
        """Get time until next request can be made."""
        if self.can_make_request():
            return 0.0
        
        if not self.requests:
            return 0.0
        
        oldest_request = self.requests[0]
        return (oldest_request + self.time_window) - time.time()


class LLMManager:
    """Manager for multiple LLM providers with load balancing and failover."""
    
    def __init__(self):
        """
          Init   function implementation.
        """
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_stats: Dict[str, ProviderStats] = defaultdict(ProviderStats)
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.cache = LLMCache() if settings.enable_llm_cache else None
        self.current_provider = settings.default_llm_provider
        self.fallback_providers = settings.fallback_providers.copy()
        self._lock = asyncio.Lock()
        
        # Initialize providers
        self._initialize_providers()
        
        # Initialize rate limiters
        if settings.enable_rate_limiting:
            self._initialize_rate_limiters()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        provider_configs = {
            LLMProvider.OPENAI: {
                'model_name': settings.default_model if settings.default_llm_provider == 'openai' else 'gpt-3.5-turbo',
                **settings.get_llm_config('openai')
            },
            LLMProvider.ANTHROPIC: {
                'model_name': settings.default_model if settings.default_llm_provider == 'anthropic' else 'claude-3-sonnet-20240229',
                **settings.get_llm_config('anthropic')
            },
            LLMProvider.HUGGINGFACE: {
                'model_name': settings.default_model if settings.default_llm_provider == 'huggingface' else 'microsoft/DialoGPT-medium',
                **settings.get_llm_config('huggingface')
            },
            LLMProvider.LOCAL: {
                'model_name': settings.default_model if settings.default_llm_provider == 'local' else 'llama2',
                **settings.get_llm_config('local')
            }
        }
        
        for provider_type, config in provider_configs.items():
            try:
                # Only initialize if API key is available (except for local)
                if provider_type == LLMProvider.LOCAL or config.get('api_key'):
                    provider = create_provider(provider_type, **config)
                    self.providers[provider_type] = provider
                    logger.info(f"Initialized {provider_type} provider")
                else:
                    logger.warning(f"Skipping {provider_type} provider - no API key configured")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type} provider: {e}")
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each provider."""
        # Different rate limits for different providers
        rate_limits = {
            LLMProvider.OPENAI: (settings.requests_per_minute, 60),
            LLMProvider.ANTHROPIC: (settings.requests_per_minute, 60),
            LLMProvider.HUGGINGFACE: (settings.requests_per_minute // 2, 60),  # More conservative
            LLMProvider.LOCAL: (settings.requests_per_minute * 2, 60),  # More generous for local
        }
        
        for provider_type, (max_requests, time_window) in rate_limits.items():
            if provider_type in self.providers:
                self.rate_limiters[provider_type] = RateLimiter(max_requests, time_window)
    
    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Generate response using the specified or best available provider."""
        
        # Check cache first
        if use_cache and self.cache:
            cache_key = self.cache.generate_cache_key(prompt, provider, model_name, temperature, max_tokens)
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Determine provider to use
        target_provider = provider or self.current_provider
        
        # Try primary provider first
        try:
            response = await self._generate_with_provider(
                target_provider, prompt, model_name, max_tokens, temperature, **kwargs
            )
            
            # Cache successful response
            if use_cache and self.cache:
                await self.cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.warning(f"Primary provider {target_provider} failed: {e}")
            
            # Try fallback providers if enabled
            if settings.enable_model_fallback and target_provider in self.fallback_providers:
                fallback_list = [p for p in self.fallback_providers if p != target_provider]
                
                for fallback_provider in fallback_list:
                    try:
                        logger.info(f"Trying fallback provider: {fallback_provider}")
                        response = await self._generate_with_provider(
                            fallback_provider, prompt, model_name, max_tokens, temperature, **kwargs
                        )
                        
                        # Cache successful response
                        if use_cache and self.cache:
                            await self.cache.set(cache_key, response)
                        
                        return response
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback provider {fallback_provider} failed: {fallback_error}")
                        continue
            
            # All providers failed
            raise LLMError(f"All providers failed. Last error: {str(e)}")
    
    async def _generate_with_provider(
        self,
        provider_type: str,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with a specific provider."""
        
        if provider_type not in self.providers:
            raise LLMError(f"Provider {provider_type} not available")
        
        provider = self.providers[provider_type]
        stats = self.provider_stats[provider_type]
        
        # Check if provider is healthy
        if not stats.is_healthy:
            # Check if enough time has passed to retry
            if (stats.last_request_time and 
                datetime.now() - stats.last_request_time < timedelta(minutes=5)):
                raise LLMError(f"Provider {provider_type} is marked as unhealthy")
        
        # Check rate limiting
        if settings.enable_rate_limiting and provider_type in self.rate_limiters:
            rate_limiter = self.rate_limiters[provider_type]
            
            if not rate_limiter.can_make_request():
                wait_time = rate_limiter.time_until_next_request()
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded for {provider_type}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            rate_limiter.record_request()
        
        # Update stats
        async with self._lock:
            stats.total_requests += 1
            stats.last_request_time = datetime.now()
        
        start_time = time.time()
        
        try:
            # Generate response
            response = await provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Update success stats
            response_time = time.time() - start_time
            async with self._lock:
                stats.successful_requests += 1
                stats.total_response_time += response_time
                stats.consecutive_failures = 0
                stats.is_healthy = True
            
            logger.debug(f"Generated response with {provider_type} in {response_time:.2f}s")
            return response
            
        except Exception as e:
            # Update failure stats
            async with self._lock:
                stats.failed_requests += 1
                stats.consecutive_failures += 1
                
                # Mark as unhealthy after consecutive failures
                if stats.consecutive_failures >= 3:
                    stats.is_healthy = False
                    logger.error(f"Marking {provider_type} as unhealthy after {stats.consecutive_failures} failures")
            
            raise e
    
    async def generate_stream(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response."""
        target_provider = provider or self.current_provider
        
        if target_provider not in self.providers:
            raise LLMError(f"Provider {target_provider} not available")
        
        provider_instance = self.providers[target_provider]
        
        # Check rate limiting
        if settings.enable_rate_limiting and target_provider in self.rate_limiters:
            rate_limiter = self.rate_limiters[target_provider]
            
            if not rate_limiter.can_make_request():
                wait_time = rate_limiter.time_until_next_request()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            rate_limiter.record_request()
        
        # Generate streaming response
        async for chunk in provider_instance.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ):
            yield chunk
    
    async def switch_provider(self, provider_type: str) -> bool:
        """Switch to a different provider."""
        if provider_type not in self.providers:
            logger.error(f"Cannot switch to unavailable provider: {provider_type}")
            return False
        
        # Test provider health
        provider = self.providers[provider_type]
        is_healthy = await provider.health_check()
        
        if not is_healthy:
            logger.error(f"Cannot switch to unhealthy provider: {provider_type}")
            return False
        
        self.current_provider = provider_type
        logger.info(f"Switched to provider: {provider_type}")
        return True
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available and healthy providers."""
        available = []
        
        for provider_type, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                if is_healthy:
                    available.append(provider_type)
            except Exception as e:
                logger.warning(f"Health check failed for {provider_type}: {e}")
        
        return available
    
    async def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        stats = {}
        
        for provider_type, provider_stats in self.provider_stats.items():
            stats[provider_type] = {
                'total_requests': provider_stats.total_requests,
                'successful_requests': provider_stats.successful_requests,
                'failed_requests': provider_stats.failed_requests,
                'success_rate': provider_stats.success_rate,
                'average_response_time': provider_stats.average_response_time,
                'is_healthy': provider_stats.is_healthy,
                'consecutive_failures': provider_stats.consecutive_failures,
                'last_request_time': provider_stats.last_request_time.isoformat() if provider_stats.last_request_time else None
            }
        
        return stats
    
    async def get_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available models for providers."""
        models = {}
        
        providers_to_check = [provider] if provider else list(self.providers.keys())
        
        for provider_type in providers_to_check:
            if provider_type in self.providers:
                try:
                    provider_models = await self.providers[provider_type].get_available_models()
                    models[provider_type] = provider_models
                except Exception as e:
                    logger.error(f"Failed to get models for {provider_type}: {e}")
                    models[provider_type] = []
        
        return models
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all providers."""
        health_status = {}
        
        for provider_type, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_status[provider_type] = is_healthy
                
                # Update stats
                async with self._lock:
                    self.provider_stats[provider_type].is_healthy = is_healthy
                    if is_healthy:
                        self.provider_stats[provider_type].consecutive_failures = 0
                        
            except Exception as e:
                logger.error(f"Health check failed for {provider_type}: {e}")
                health_status[provider_type] = False
                
                async with self._lock:
                    self.provider_stats[provider_type].is_healthy = False
        
        return health_status
    
    def get_best_provider(self) -> str:
        """Get the best performing provider based on stats."""
        if not self.provider_stats:
            return self.current_provider
        
        best_provider = None
        best_score = -1
        
        for provider_type, stats in self.provider_stats.items():
            if not stats.is_healthy or provider_type not in self.providers:
                continue
            
            # Calculate score based on success rate and response time
            success_rate = stats.success_rate
            avg_response_time = stats.average_response_time
            
            # Normalize response time (lower is better)
            time_score = 1.0 / (1.0 + avg_response_time) if avg_response_time > 0 else 1.0
            
            # Combined score (weighted)
            score = (success_rate * 0.7) + (time_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_provider = provider_type
        
        return best_provider or self.current_provider
    
    async def auto_select_provider(self) -> str:
        """Automatically select the best provider and switch to it."""
        best_provider = self.get_best_provider()
        
        if best_provider != self.current_provider:
            success = await self.switch_provider(best_provider)
            if success:
                logger.info(f"Auto-switched to best provider: {best_provider}")
                return best_provider
        
        return self.current_provider
    
    def reset_stats(self, provider: Optional[str] = None):
        """Reset statistics for a provider or all providers."""
        if provider:
            if provider in self.provider_stats:
                self.provider_stats[provider] = ProviderStats()
        else:
            self.provider_stats.clear()
        
        logger.info(f"Reset stats for {provider or 'all providers'}")
    
    async def close(self):
        """Clean up resources."""
        if self.cache:
            await self.cache.close()
        
        # Close provider connections if needed
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()


# Global LLM manager instance
llm_manager = LLMManager()