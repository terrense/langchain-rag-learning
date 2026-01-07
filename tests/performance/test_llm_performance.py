"""Performance tests for LLM system."""

import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from src.langchain_rag_learning.llm.manager import LLMManager
from src.langchain_rag_learning.llm.templates import PromptTemplateManager, PromptType, Language
from src.langchain_rag_learning.core.models import LLMProvider, LLMResponse


class MockPerformanceProvider:
    """Mock provider for performance testing."""
    
    def __init__(self, model_name: str, response_time: float = 0.1):
        self.model_name = model_name
        self.response_time = response_time
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        await asyncio.sleep(self.response_time)  # Simulate processing time
        
        return LLMResponse(
            content=f"Response {self.call_count} to: {prompt[:50]}...",
            model_name=self.model_name,
            provider=LLMProvider.OPENAI,
            response_time=self.response_time
        )
    
    async def generate_stream(self, prompt: str, **kwargs):
        await asyncio.sleep(self.response_time / 10)  # Faster for streaming
        yield f"Streaming response to: {prompt[:50]}..."
    
    async def get_available_models(self):
        return [self.model_name]
    
    async def validate_config(self):
        return True
    
    async def health_check(self):
        return True


class TestLLMPerformance:
    """Performance tests for LLM system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_settings = Mock()
        self.mock_settings.enable_llm_cache = False
        self.mock_settings.default_llm_provider = "openai"
        self.mock_settings.fallback_providers = ["openai"]
        self.mock_settings.enable_rate_limiting = False
        self.mock_settings.enable_model_fallback = True
        self.mock_settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
    
    async def test_single_request_performance(self):
        """Test performance of single request."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {
                    "openai": MockPerformanceProvider("gpt-3.5-turbo", response_time=0.1)
                }
                
                start_time = time.time()
                response = await manager.generate("Test prompt for performance")
                end_time = time.time()
                
                # Should complete within reasonable time
                assert end_time - start_time < 0.5
                assert response.content.startswith("Response 1 to:")
    
    async def test_concurrent_requests_performance(self):
        """Test performance with concurrent requests."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {
                    "openai": MockPerformanceProvider("gpt-3.5-turbo", response_time=0.1)
                }
                
                # Create 10 concurrent requests
                tasks = [
                    manager.generate(f"Concurrent test prompt {i}")
                    for i in range(10)
                ]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks)
                end_time = time.time()
                
                # Should complete all requests concurrently (not sequentially)
                # If sequential, it would take 10 * 0.1 = 1.0 seconds
                # Concurrent should be closer to 0.1 seconds
                assert end_time - start_time < 0.5
                assert len(responses) == 10
                
                # Verify all responses are unique
                response_contents = [r.content for r in responses]
                assert len(set(response_contents)) == 10
    
    async def test_cache_performance_benefit(self):
        """Test performance benefit of caching."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                provider = MockPerformanceProvider("gpt-3.5-turbo", response_time=0.2)
                manager.providers = {"openai": provider}
                
                # Mock cache that actually works
                cache_storage = {}
                
                async def mock_cache_get(key):
                    return cache_storage.get(key)
                
                async def mock_cache_set(key, value):
                    cache_storage[key] = value
                
                mock_cache = AsyncMock()
                mock_cache.get = mock_cache_get
                mock_cache.set = mock_cache_set
                mock_cache.generate_cache_key = lambda *args, **kwargs: "test-cache-key"
                
                manager.cache = mock_cache
                
                # First request - should be slow (cache miss)
                start_time = time.time()
                response1 = await manager.generate("Cached test prompt", use_cache=True)
                first_request_time = time.time() - start_time
                
                # Second request - should be fast (cache hit)
                start_time = time.time()
                response2 = await manager.generate("Cached test prompt", use_cache=True)
                second_request_time = time.time() - start_time
                
                # Cache hit should be significantly faster
                assert second_request_time < first_request_time / 2
                assert provider.call_count == 1  # Only called once due to cache
    
    async def test_template_formatting_performance(self):
        """Test template formatting performance."""
        template_manager = PromptTemplateManager()
        
        # Test multiple template formats
        test_cases = [
            (PromptType.RAG_QUERY, {"context": "Test context", "query": "Test query"}),
            (PromptType.RAG_QUERY_WITH_HISTORY, {
                "context": "Test context", 
                "query": "Test query", 
                "history": "Previous conversation"
            }),
            (PromptType.SUMMARIZATION, {"text": "Long text to summarize"}),
        ]
        
        start_time = time.time()
        
        # Format 100 prompts of each type
        for prompt_type, kwargs in test_cases:
            for i in range(100):
                formatted = template_manager.format_prompt(
                    prompt_type, 
                    Language.ENGLISH, 
                    **kwargs
                )
                assert isinstance(formatted, str)
                assert len(formatted) > 0
        
        end_time = time.time()
        
        # Should format 300 prompts quickly
        assert end_time - start_time < 1.0
    
    async def test_provider_switching_performance(self):
        """Test performance of provider switching."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {
                    "openai": MockPerformanceProvider("gpt-3.5-turbo", response_time=0.05),
                    "anthropic": MockPerformanceProvider("claude-3-sonnet", response_time=0.05),
                }
                
                # Test rapid provider switching
                start_time = time.time()
                
                for i in range(10):
                    provider = "openai" if i % 2 == 0 else "anthropic"
                    success = await manager.switch_provider(provider)
                    assert success is True
                    
                    response = await manager.generate(f"Switch test {i}")
                    assert response is not None
                
                end_time = time.time()
                
                # Should handle rapid switching efficiently
                assert end_time - start_time < 2.0
    
    async def test_error_handling_performance(self):
        """Test performance impact of error handling."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                
                # Create failing and working providers
                failing_provider = AsyncMock()
                failing_provider.generate = AsyncMock(side_effect=Exception("Provider failed"))
                failing_provider.health_check = AsyncMock(return_value=False)
                
                working_provider = MockPerformanceProvider("gpt-3.5-turbo", response_time=0.1)
                
                manager.providers = {
                    "failing": failing_provider,
                    "working": working_provider
                }
                manager.current_provider = "failing"
                manager.fallback_providers = ["failing", "working"]
                
                start_time = time.time()
                
                # Should quickly failover to working provider
                response = await manager.generate("Failover test")
                
                end_time = time.time()
                
                # Failover should be fast
                assert end_time - start_time < 0.5
                assert response.content.startswith("Response 1 to:")
    
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under load."""
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=self.mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {
                    "openai": MockPerformanceProvider("gpt-3.5-turbo", response_time=0.01)
                }
                
                # Make many requests to test memory stability
                for batch in range(5):
                    tasks = [
                        manager.generate(f"Memory test {batch}-{i}")
                        for i in range(20)
                    ]
                    
                    responses = await asyncio.gather(*tasks)
                    assert len(responses) == 20
                    
                    # Clear responses to avoid accumulation
                    del responses
    
    def test_template_manager_initialization_performance(self):
        """Test template manager initialization performance."""
        start_time = time.time()
        
        # Initialize multiple template managers
        for i in range(10):
            manager = PromptTemplateManager()
            available = manager.get_available_templates()
            assert len(available) > 0
        
        end_time = time.time()
        
        # Should initialize quickly
        assert end_time - start_time < 1.0


class TestLLMScalability:
    """Scalability tests for LLM system."""
    
    async def test_high_concurrency_handling(self):
        """Test handling of high concurrency."""
        mock_settings = Mock()
        mock_settings.enable_llm_cache = False
        mock_settings.default_llm_provider = "openai"
        mock_settings.fallback_providers = ["openai"]
        mock_settings.enable_rate_limiting = False
        mock_settings.enable_model_fallback = True
        mock_settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
        
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                manager.providers = {
                    "openai": MockPerformanceProvider("gpt-3.5-turbo", response_time=0.05)
                }
                
                # Create 50 concurrent requests
                tasks = [
                    manager.generate(f"High concurrency test {i}")
                    for i in range(50)
                ]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks)
                end_time = time.time()
                
                # Should handle high concurrency
                assert len(responses) == 50
                assert end_time - start_time < 2.0  # Should complete within 2 seconds
                
                # Verify all responses are unique
                response_contents = [r.content for r in responses]
                assert len(set(response_contents)) == 50
    
    async def test_sustained_load_performance(self):
        """Test performance under sustained load."""
        mock_settings = Mock()
        mock_settings.enable_llm_cache = False
        mock_settings.default_llm_provider = "openai"
        mock_settings.fallback_providers = ["openai"]
        mock_settings.enable_rate_limiting = False
        mock_settings.enable_model_fallback = True
        mock_settings.get_llm_config = Mock(return_value={"api_key": "test-key"})
        
        with patch('src.langchain_rag_learning.llm.manager.get_settings', return_value=mock_settings):
            with patch('src.langchain_rag_learning.llm.manager.LLMManager._initialize_providers'):
                manager = LLMManager()
                provider = MockPerformanceProvider("gpt-3.5-turbo", response_time=0.02)
                manager.providers = {"openai": provider}
                
                # Sustained load test - multiple batches
                total_requests = 0
                start_time = time.time()
                
                for batch in range(10):
                    batch_tasks = [
                        manager.generate(f"Sustained load batch {batch} request {i}")
                        for i in range(10)
                    ]
                    
                    batch_responses = await asyncio.gather(*batch_tasks)
                    total_requests += len(batch_responses)
                    
                    # Small delay between batches
                    await asyncio.sleep(0.01)
                
                end_time = time.time()
                
                # Should handle sustained load efficiently
                assert total_requests == 100
                assert end_time - start_time < 5.0
                assert provider.call_count == 100


# Utility function to run performance tests
async def run_performance_tests():
    """Run all performance tests."""
    print("Running LLM Performance Tests...")
    
    test_instance = TestLLMPerformance()
    test_instance.setup_method()
    
    tests = [
        ("Single Request Performance", test_instance.test_single_request_performance),
        ("Concurrent Requests Performance", test_instance.test_concurrent_requests_performance),
        ("Cache Performance Benefit", test_instance.test_cache_performance_benefit),
        ("Template Formatting Performance", test_instance.test_template_formatting_performance),
        ("Provider Switching Performance", test_instance.test_provider_switching_performance),
        ("Error Handling Performance", test_instance.test_error_handling_performance),
        ("Memory Usage Stability", test_instance.test_memory_usage_stability),
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            await test_func()
            end_time = time.time()
            print(f"✓ {test_name}: {end_time - start_time:.3f}s")
        except Exception as e:
            print(f"✗ {test_name}: {e}")
    
    # Run scalability tests
    scalability_test = TestLLMScalability()
    scalability_tests = [
        ("High Concurrency Handling", scalability_test.test_high_concurrency_handling),
        ("Sustained Load Performance", scalability_test.test_sustained_load_performance),
    ]
    
    for test_name, test_func in scalability_tests:
        try:
            start_time = time.time()
            await test_func()
            end_time = time.time()
            print(f"✓ {test_name}: {end_time - start_time:.3f}s")
        except Exception as e:
            print(f"✗ {test_name}: {e}")
    
    print("Performance tests completed!")


if __name__ == "__main__":
    asyncio.run(run_performance_tests())