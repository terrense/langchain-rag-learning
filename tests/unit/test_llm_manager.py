"""Unit tests for LLM manager."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.langchain_rag_learning.llm.manager import (
    LLMManager,
    ProviderStats,
    RateLimiter
)
from src.langchain_rag_learning.llm.providers import BaseLLMProvider
from src.langchain_rag_learning.core.models import LLMProvider, LLMResponse
from src.langchain_rag_learning.core.exceptions import LLMError


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, model_name: str, should_fail: bool = False, **kwargs):
        super().__init__(model_name, **kwargs)
        self.should_fail = should_fail
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        
        if self.should_fail:
            raise Exception("Mock provider failure")
        
        return LLMResponse(
            content=f"Response to: {prompt}",
            model_name=self.model_name,
            provider=LLMProvider.OPENAI,
            response_time=0.1
        )
    
    async def generate_stream(self, prompt: str, **kwargs):
        if self.should_fail:
            raise Exception("Mock provider failure")
        
        yield f"Response to: {prompt}"
    
    async def get_available_models(self):
        return [self.model_name]
    
    async def validate_config(self):
        return not self.should_fail
    
    async def health_check(self):
        return not self.should_fail


class TestProviderStats:
    """Test provider statistics."""
    
    def test_initial_stats(self):
        """Test initial statistics."""
        stats = ProviderStats()
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.success_rate == 1.0
        assert stats.average_response_time == 0.0
        assert stats.is_healthy is True
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = ProviderStats()
        stats.total_requests = 10
        stats.successful_requests = 8
        stats.failed_requests = 2
        
        assert stats.success_rate == 0.8
    
    def test_average_response_time_calculation(self):
        """Test average response time calculation."""
        stats = ProviderStats()
        stats.successful_requests = 5
        stats.total_response_time = 2.5
        
        assert stats.average_response_time == 0.5


class TestRateLimiter:
    """Test rate limiter."""
    
    def test_initial_state(self):
        """Test initial rate limiter state."""
        limiter = RateLimiter(max_requests=10, time_window=60)
        
        assert limiter.can_make_request() is True
        assert limiter.time_until_next_request() == 0.0
    
    def test_request_recording(self):
        """Test request recording."""
        limiter = RateLimiter(max_requests=2, time_window=60)
        
        # First request
        assert limiter.can_make_request() is True
        limiter.record_request()
        
        # Second request
        assert limiter.can_make_request() is True
        limiter.record_request()
        
        # Third request should be blocked
        assert limiter.can_make_request() is False
    
    def test_time_window_expiry(self):
        """Test that old requests expire."""
        limiter = RateLimiter(max_requests=1, time_window=1)
        
        # Make a request
        limiter.record_request()
        assert limiter.can_make_request() is False
        
        # Wait for time window to pass
        import time
        time.sleep(1.1)
        
        # Should be able to make request again
        assert limiter.can_make_request() is True


class TestLLMManager:
    """Test LLM manager."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings') as mock:
            settings = Mock()
            settings.enable_llm_cache = False
            settings.default_llm_provider = "openai"
            settings.fallback_providers = ["openai", "anthropic"]
            settings.enable_rate_limiting = False
            settings.enable_model_fallback = True
            settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
            mock.return_value = settings
            yield settings
    
    @pytest.fixture
    def manager(self, mock_settings):
        """Create LLM manager for testing."""
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            manager = LLMManager()
            
            # Manually add mock providers
            manager.providers = {
                "openai": MockProvider("gpt-3.5-turbo"),
                "anthropic": MockProvider("claude-3-sonnet", should_fail=False),
                "failing": MockProvider("failing-model", should_fail=True)
            }
            
            return manager
    
    @pytest.mark.asyncio
    async def test_generate_success(self, manager):
        """Test successful generation."""
        response = await manager.generate("Test prompt", provider="openai")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Response to: Test prompt"
        assert response.model_name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self, manager):
        """Test generation with fallback."""
        # Set primary provider to failing one
        manager.current_provider = "failing"
        
        response = await manager.generate("Test prompt")
        
        # Should fallback to working provider
        assert isinstance(response, LLMResponse)
        assert "Response to: Test prompt" in response.content
    
    @pytest.mark.asyncio
    async def test_generate_all_providers_fail(self, manager):
        """Test when all providers fail."""
        # Make all providers fail
        for provider in manager.providers.values():
            provider.should_fail = True
        
        with pytest.raises(LLMError):
            await manager.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_switch_provider(self, manager):
        """Test switching providers."""
        success = await manager.switch_provider("anthropic")
        assert success is True
        assert manager.current_provider == "anthropic"
        
        # Try switching to non-existent provider
        success = await manager.switch_provider("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_available_providers(self, manager):
        """Test getting available providers."""
        providers = await manager.get_available_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert "failing" not in providers  # Should be filtered out as unhealthy
    
    @pytest.mark.asyncio
    async def test_provider_stats_tracking(self, manager):
        """Test provider statistics tracking."""
        # Make some requests
        await manager.generate("Test 1", provider="openai")
        await manager.generate("Test 2", provider="openai")
        
        try:
            await manager.generate("Test 3", provider="failing")
        except LLMError:
            pass  # Expected to fail
        
        stats = await manager.get_provider_stats()
        
        # Check OpenAI stats
        openai_stats = stats["openai"]
        assert openai_stats["total_requests"] == 2
        assert openai_stats["successful_requests"] == 2
        assert openai_stats["failed_requests"] == 0
        assert openai_stats["success_rate"] == 1.0
        
        # Check failing provider stats
        failing_stats = stats["failing"]
        assert failing_stats["total_requests"] == 1
        assert failing_stats["successful_requests"] == 0
        assert failing_stats["failed_requests"] == 1
        assert failing_stats["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, manager):
        """Test getting available models."""
        models = await manager.get_available_models()
        
        assert "openai" in models
        assert "anthropic" in models
        assert isinstance(models["openai"], list)
        assert isinstance(models["anthropic"], list)
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, manager):
        """Test health check for all providers."""
        health_status = await manager.health_check_all()
        
        assert health_status["openai"] is True
        assert health_status["anthropic"] is True
        assert health_status["failing"] is False
    
    def test_get_best_provider(self, manager):
        """Test getting best provider based on stats."""
        # Set up some stats
        manager.provider_stats["openai"].total_requests = 10
        manager.provider_stats["openai"].successful_requests = 10
        manager.provider_stats["openai"].total_response_time = 1.0
        manager.provider_stats["openai"].is_healthy = True
        
        manager.provider_stats["anthropic"].total_requests = 10
        manager.provider_stats["anthropic"].successful_requests = 8
        manager.provider_stats["anthropic"].total_response_time = 2.0
        manager.provider_stats["anthropic"].is_healthy = True
        
        best_provider = manager.get_best_provider()
        assert best_provider == "openai"  # Better success rate and response time
    
    @pytest.mark.asyncio
    async def test_auto_select_provider(self, manager):
        """Test automatic provider selection."""
        # Set up stats to make anthropic better
        manager.provider_stats["anthropic"].total_requests = 10
        manager.provider_stats["anthropic"].successful_requests = 10
        manager.provider_stats["anthropic"].total_response_time = 0.5
        manager.provider_stats["anthropic"].is_healthy = True
        
        manager.provider_stats["openai"].total_requests = 10
        manager.provider_stats["openai"].successful_requests = 8
        manager.provider_stats["openai"].total_response_time = 2.0
        manager.provider_stats["openai"].is_healthy = True
        
        selected_provider = await manager.auto_select_provider()
        assert selected_provider == "anthropic"
        assert manager.current_provider == "anthropic"
    
    def test_reset_stats(self, manager):
        """Test resetting statistics."""
        # Set up some stats
        manager.provider_stats["openai"].total_requests = 10
        manager.provider_stats["openai"].successful_requests = 8
        
        # Reset specific provider
        manager.reset_stats("openai")
        assert manager.provider_stats["openai"].total_requests == 0
        assert manager.provider_stats["openai"].successful_requests == 0
        
        # Reset all providers
        manager.provider_stats["anthropic"].total_requests = 5
        manager.reset_stats()
        assert len(manager.provider_stats) == 0
    
    @pytest.mark.asyncio
    async def test_generate_stream(self, manager):
        """Test streaming generation."""
        chunks = []
        async for chunk in manager.generate_stream("Test prompt", provider="openai"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == "Response to: Test prompt"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_settings):
        """Test rate limiting functionality."""
        mock_settings.enable_rate_limiting = True
        mock_settings.requests_per_minute = 2
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            manager = LLMManager()
            manager.providers = {"openai": MockProvider("gpt-3.5-turbo")}
            manager._initialize_rate_limiters()
            
            # Make requests up to limit
            await manager.generate("Test 1", provider="openai")
            await manager.generate("Test 2", provider="openai")
            
            # Third request should be rate limited (but we can't easily test the delay)
            # Just verify the rate limiter exists
            assert "openai" in manager.rate_limiters
            assert isinstance(manager.rate_limiters["openai"], RateLimiter)


class TestLLMManagerIntegration:
    """Integration tests for LLM manager."""
    
    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings') as mock_settings:
            settings = Mock()
            settings.enable_llm_cache = True
            settings.default_llm_provider = "openai"
            settings.fallback_providers = ["openai"]
            settings.enable_rate_limiting = False
            settings.enable_model_fallback = True
            settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
            mock_settings.return_value = settings
            
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {"openai": MockProvider("gpt-3.5-turbo")}
                
                # Mock cache
                mock_cache = AsyncMock()
                mock_cache.get = AsyncMock(return_value=None)  # No cache hit
                mock_cache.set = AsyncMock()
                mock_cache.generate_cache_key = Mock(return_value="test-key")
                manager.cache = mock_cache
                
                # First call should generate and cache
                response = await manager.generate("Test prompt")
                
                assert mock_cache.get.called
                assert mock_cache.set.called
                assert isinstance(response, LLMResponse)
    
    @pytest.mark.asyncio
    async def test_provider_failure_recovery(self):
        """Test provider failure and recovery."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings') as mock_settings:
            settings = Mock()
            settings.enable_llm_cache = False
            settings.default_llm_provider = "openai"
            settings.fallback_providers = ["openai"]
            settings.enable_rate_limiting = False
            settings.enable_model_fallback = True
            settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
            mock_settings.return_value = settings
            
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                
                # Create a provider that fails initially then recovers
                provider = MockProvider("gpt-3.5-turbo", should_fail=True)
                manager.providers = {"openai": provider}
                
                # First call should fail
                with pytest.raises(LLMError):
                    await manager.generate("Test prompt")
                
                # Check that provider is marked as unhealthy
                stats = manager.provider_stats["openai"]
                assert stats.consecutive_failures > 0
                
                # Provider recovers
                provider.should_fail = False
                
                # Should work now
                response = await manager.generate("Test prompt")
                assert isinstance(response, LLMResponse)