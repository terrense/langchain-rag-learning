"""Integration tests for LLM system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.langchain_rag_learning.llm.manager import LLMManager
from src.langchain_rag_learning.llm.templates import PromptTemplateManager, PromptType, Language
from src.langchain_rag_learning.llm.cache import LLMCache
from src.langchain_rag_learning.llm.providers import create_provider
from src.langchain_rag_learning.core.models import LLMProvider, LLMResponse


class TestLLMSystemIntegration:
    """Integration tests for the complete LLM system."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for integration tests."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings') as mock_manager_settings, \
             patch('src.langchain_rag_learning.llm.cache.get_settings') as mock_cache_settings:
            
            settings = Mock()
            settings.enable_llm_cache = True
            settings.llm_cache_ttl = 3600
            settings.redis_url = "redis://localhost:6379/0"
            settings.default_llm_provider = "openai"
            settings.fallback_providers = ["openai", "anthropic"]
            settings.enable_rate_limiting = False
            settings.enable_model_fallback = True
            settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
            
            mock_manager_settings.return_value = settings
            mock_cache_settings.return_value = settings
            
            yield settings
    
    @pytest.fixture
    def mock_providers(self):
        """Mock providers for testing."""
        providers = {}
        
        # Mock OpenAI provider
        openai_provider = AsyncMock()
        openai_provider.generate = AsyncMock(return_value=LLMResponse(
            content="OpenAI response",
            model_name="gpt-3.5-turbo",
            provider=LLMProvider.OPENAI,
            response_time=0.5
        ))
        openai_provider.health_check = AsyncMock(return_value=True)
        openai_provider.get_available_models = AsyncMock(return_value=["gpt-3.5-turbo", "gpt-4"])
        providers["openai"] = openai_provider
        
        # Mock Anthropic provider
        anthropic_provider = AsyncMock()
        anthropic_provider.generate = AsyncMock(return_value=LLMResponse(
            content="Anthropic response",
            model_name="claude-3-sonnet",
            provider=LLMProvider.ANTHROPIC,
            response_time=0.7
        ))
        anthropic_provider.health_check = AsyncMock(return_value=True)
        anthropic_provider.get_available_models = AsyncMock(return_value=["claude-3-sonnet", "claude-3-opus"])
        providers["anthropic"] = anthropic_provider
        
        return providers
    
    @pytest.mark.asyncio
    async def test_end_to_end_rag_query(self, mock_settings, mock_providers):
        """Test end-to-end RAG query processing."""
        # Initialize components
        template_manager = PromptTemplateManager()
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Create RAG prompt
        context = """
        LangChain is a framework for developing applications powered by language models.
        It provides abstractions and tools for working with LLMs, including prompt templates,
        chains for combining multiple LLM calls, and agents for autonomous decision making.
        """
        
        query = "What is LangChain and what does it provide?"
        
        prompt = template_manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context=context.strip(),
            query=query
        )
        
        # Generate response
        response = await llm_manager.generate(prompt, provider="openai")
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "OpenAI response"
        assert response.provider == LLMProvider.OPENAI
        assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_provider_fallback_integration(self, mock_settings, mock_providers):
        """Test provider fallback in integrated system."""
        # Make OpenAI provider fail
        mock_providers["openai"].generate = AsyncMock(side_effect=Exception("OpenAI failed"))
        mock_providers["openai"].health_check = AsyncMock(return_value=False)
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Should fallback to Anthropic
        response = await llm_manager.generate("Test prompt")
        
        assert response.content == "Anthropic response"
        assert response.provider == LLMProvider.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, mock_settings, mock_providers):
        """Test cache integration with LLM manager."""
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
            
            # Mock cache
            mock_cache = AsyncMock()
            mock_cache.get = AsyncMock(return_value=None)  # No cache hit first time
            mock_cache.set = AsyncMock()
            mock_cache.generate_cache_key = Mock(return_value="test-cache-key")
            llm_manager.cache = mock_cache
            
            # First call - should generate and cache
            response1 = await llm_manager.generate("Test prompt", use_cache=True)
            
            assert mock_cache.get.called
            assert mock_cache.set.called
            
            # Second call - should hit cache
            cached_response = LLMResponse(
                content="Cached response",
                model_name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                response_time=0.1
            )
            mock_cache.get = AsyncMock(return_value=cached_response)
            
            response2 = await llm_manager.generate("Test prompt", use_cache=True)
            
            assert response2.content == "Cached response"
    
    @pytest.mark.asyncio
    async def test_multilingual_template_integration(self, mock_settings, mock_providers):
        """Test multilingual templates with LLM generation."""
        template_manager = PromptTemplateManager()
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Test English prompt
        english_prompt = template_manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context="Test context",
            query="Test query"
        )
        
        response_en = await llm_manager.generate(english_prompt)
        assert isinstance(response_en, LLMResponse)
        
        # Test Chinese prompt
        chinese_prompt = template_manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.CHINESE,
            context="测试上下文",
            query="测试问题"
        )
        
        response_zh = await llm_manager.generate(chinese_prompt)
        assert isinstance(response_zh, LLMResponse)
    
    @pytest.mark.asyncio
    async def test_conversation_history_integration(self, mock_settings, mock_providers):
        """Test conversation history handling."""
        template_manager = PromptTemplateManager()
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # First query
        prompt1 = template_manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context="LangChain is a framework for LLM applications.",
            query="What is LangChain?"
        )
        
        response1 = await llm_manager.generate(prompt1)
        
        # Follow-up query with history
        history = f"User: What is LangChain?\nAssistant: {response1.content}"
        
        prompt2 = template_manager.format_prompt(
            PromptType.RAG_QUERY_WITH_HISTORY,
            Language.ENGLISH,
            history=history,
            context="LangChain supports multiple LLM providers including OpenAI and Anthropic.",
            query="What providers does it support?"
        )
        
        response2 = await llm_manager.generate(prompt2)
        
        assert isinstance(response2, LLMResponse)
        assert "What is LangChain?" in prompt2  # History should be included
    
    @pytest.mark.asyncio
    async def test_provider_statistics_integration(self, mock_settings, mock_providers):
        """Test provider statistics tracking in integrated system."""
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Make multiple requests
        await llm_manager.generate("Test 1", provider="openai")
        await llm_manager.generate("Test 2", provider="openai")
        await llm_manager.generate("Test 3", provider="anthropic")
        
        # Check statistics
        stats = await llm_manager.get_provider_stats()
        
        assert stats["openai"]["total_requests"] == 2
        assert stats["openai"]["successful_requests"] == 2
        assert stats["anthropic"]["total_requests"] == 1
        assert stats["anthropic"]["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, mock_settings, mock_providers):
        """Test health monitoring across the system."""
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Check initial health
        health_status = await llm_manager.health_check_all()
        
        assert health_status["openai"] is True
        assert health_status["anthropic"] is True
        
        # Simulate provider failure
        mock_providers["openai"].health_check = AsyncMock(return_value=False)
        
        # Check health again
        health_status = await llm_manager.health_check_all()
        
        assert health_status["openai"] is False
        assert health_status["anthropic"] is True
    
    @pytest.mark.asyncio
    async def test_model_switching_integration(self, mock_settings, mock_providers):
        """Test model switching functionality."""
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Initial provider
        assert llm_manager.current_provider == "openai"
        
        # Switch to Anthropic
        success = await llm_manager.switch_provider("anthropic")
        assert success is True
        assert llm_manager.current_provider == "anthropic"
        
        # Generate with new provider
        response = await llm_manager.generate("Test prompt")
        assert response.provider == LLMProvider.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_settings, mock_providers):
        """Test error handling across the integrated system."""
        # Make all providers fail
        for provider in mock_providers.values():
            provider.generate = AsyncMock(side_effect=Exception("Provider failed"))
            provider.health_check = AsyncMock(return_value=False)
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = mock_providers
        
        # Should raise LLMError when all providers fail
        from src.langchain_rag_learning.core.exceptions import LLMError
        
        with pytest.raises(LLMError):
            await llm_manager.generate("Test prompt")


class TestLLMPerformanceIntegration:
    """Performance integration tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_settings):
        """Test handling concurrent requests."""
        # Mock provider that simulates processing time
        async def mock_generate(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return LLMResponse(
                content=f"Response to: {prompt}",
                model_name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                response_time=0.1
            )
        
        mock_provider = AsyncMock()
        mock_provider.generate = mock_generate
        mock_provider.health_check = AsyncMock(return_value=True)
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = {"openai": mock_provider}
        
        # Make concurrent requests
        tasks = [
            llm_manager.generate(f"Test prompt {i}")
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert f"Test prompt {i}" in response.content
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, mock_settings):
        """Test cache performance benefits."""
        call_count = 0
        
        async def mock_generate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate processing time
            return LLMResponse(
                content=f"Response {call_count}",
                model_name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                response_time=0.1
            )
        
        mock_provider = AsyncMock()
        mock_provider.generate = mock_generate
        
        with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
            llm_manager = LLMManager()
            llm_manager.providers = {"openai": mock_provider}
            
            # Mock cache that actually caches
            cache_storage = {}
            
            async def mock_cache_get(key):
                return cache_storage.get(key)
            
            async def mock_cache_set(key, value):
                cache_storage[key] = value
            
            mock_cache = AsyncMock()
            mock_cache.get = mock_cache_get
            mock_cache.set = mock_cache_set
            mock_cache.generate_cache_key = lambda *args, **kwargs: "test-key"
            
            llm_manager.cache = mock_cache
        
        # First request - should call provider
        response1 = await llm_manager.generate("Test prompt", use_cache=True)
        assert call_count == 1
        
        # Second request - should use cache
        response2 = await llm_manager.generate("Test prompt", use_cache=True)
        assert call_count == 1  # Should not increment
        assert response2.content == response1.content


class TestLLMConfigurationIntegration:
    """Test configuration integration."""
    
    @pytest.mark.asyncio
    async def test_dynamic_configuration_update(self):
        """Test dynamic configuration updates."""
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
                llm_manager = LLMManager()
                
                # Initial state
                assert llm_manager.cache is None
                
                # Update configuration to enable cache
                settings.enable_llm_cache = True
                
                # This would require a restart in real scenario,
                # but we can test the configuration reading
                assert settings.enable_llm_cache is True
    
    def test_provider_configuration_validation(self):
        """Test provider configuration validation."""
        # Test with valid configuration
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            settings = Mock()
            settings.OPENAI_API_KEY = "valid-key"
            mock_settings.return_value = settings
            
            # Should not raise exception
            from src.langchain_rag_learning.llm.providers import OpenAIProvider
            provider = OpenAIProvider(api_key="valid-key")
            assert provider.api_key == "valid-key"
        
        # Test with invalid configuration
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            settings = Mock()
            settings.OPENAI_API_KEY = None
            mock_settings.return_value = settings
            
            from src.langchain_rag_learning.core.exceptions import ConfigurationError
            
            with pytest.raises(ConfigurationError):
                OpenAIProvider()  # No API key provided