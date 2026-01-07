"""Unit tests for LLM providers."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.langchain_rag_learning.llm.providers import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
    LocalProvider,
    create_provider
)
from src.langchain_rag_learning.core.models import LLMProvider, LLMResponse
from src.langchain_rag_learning.core.exceptions import LLMError, ConfigurationError


class TestBaseLLMProvider:
    """Test base LLM provider functionality."""
    
    def test_base_provider_initialization(self):
        """Test base provider initialization."""
        
        class TestProvider(BaseLLMProvider):
            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content="test",
                    model_name=self.model_name,
                    provider=LLMProvider.OPENAI,
                    response_time=0.1
                )
            
            async def generate_stream(self, prompt, **kwargs):
                yield "test"
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_config(self):
                return True
        
        provider = TestProvider("test-model", param1="value1")
        
        assert provider.model_name == "test-model"
        assert provider.config == {"param1": "value1"}
        assert provider.provider_type is None
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        
        class TestProvider(BaseLLMProvider):
            async def generate(self, prompt, **kwargs):
                pass
            
            async def generate_stream(self, prompt, **kwargs):
                yield "test"
            
            async def get_available_models(self):
                return []
            
            async def validate_config(self):
                return True
        
        provider = TestProvider("test-model")
        health = await provider.health_check()
        assert health is True


class TestOpenAIProvider:
    """Test OpenAI provider."""
    
    def test_initialization_with_api_key(self):
        """Test OpenAI provider initialization with API key."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            
            provider = OpenAIProvider(api_key="test-key")
            
            assert provider.model_name == "gpt-3.5-turbo"
            assert provider.provider_type == LLMProvider.OPENAI
            assert provider.api_key == "test-key"
    
    def test_initialization_without_api_key(self):
        """Test OpenAI provider initialization without API key."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = None
            
            with pytest.raises(ConfigurationError):
                OpenAIProvider()
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test OpenAI generation."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            
            provider = OpenAIProvider(api_key="test-key")
            
            # Mock the LangChain client
            mock_client = AsyncMock()
            mock_client.predict = AsyncMock(return_value="Test response")
            provider._client = mock_client
            
            with patch('asyncio.to_thread', return_value="Test response"):
                response = await provider.generate("Test prompt")
                
                assert isinstance(response, LLMResponse)
                assert response.content == "Test response"
                assert response.model_name == "gpt-3.5-turbo"
                assert response.provider == LLMProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_generate_with_error(self):
        """Test OpenAI generation with error."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            
            provider = OpenAIProvider(api_key="test-key")
            
            # Mock the LangChain client to raise an error
            mock_client = AsyncMock()
            mock_client.predict = AsyncMock(side_effect=Exception("API Error"))
            provider._client = mock_client
            
            with patch('asyncio.to_thread', side_effect=Exception("API Error")):
                with pytest.raises(LLMError):
                    await provider.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test getting available models."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            
            provider = OpenAIProvider(api_key="test-key")
            models = await provider.get_available_models()
            
            assert isinstance(models, list)
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models


class TestAnthropicProvider:
    """Test Anthropic provider."""
    
    def test_initialization_with_api_key(self):
        """Test Anthropic provider initialization with API key."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.ANTHROPIC_API_KEY = "test-key"
            
            provider = AnthropicProvider(api_key="test-key")
            
            assert provider.model_name == "claude-3-sonnet-20240229"
            assert provider.provider_type == LLMProvider.ANTHROPIC
            assert provider.api_key == "test-key"
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test Anthropic generation."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.ANTHROPIC_API_KEY = "test-key"
            
            provider = AnthropicProvider(api_key="test-key")
            
            # Mock the LangChain client
            mock_client = AsyncMock()
            mock_client.predict = AsyncMock(return_value="Test response")
            provider._client = mock_client
            
            with patch('asyncio.to_thread', return_value="Test response"):
                response = await provider.generate("Test prompt")
                
                assert isinstance(response, LLMResponse)
                assert response.content == "Test response"
                assert response.provider == LLMProvider.ANTHROPIC


class TestHuggingFaceProvider:
    """Test HuggingFace provider."""
    
    def test_initialization_local(self):
        """Test HuggingFace provider initialization for local use."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.HUGGINGFACE_API_KEY = None
            
            with patch('transformers.pipeline') as mock_pipeline:
                mock_pipeline.return_value = Mock()
                
                provider = HuggingFaceProvider(use_local=True)
                
                assert provider.model_name == "microsoft/DialoGPT-medium"
                assert provider.provider_type == LLMProvider.HUGGINGFACE
                assert provider.use_local is True
    
    @pytest.mark.asyncio
    async def test_generate_local(self):
        """Test HuggingFace local generation."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.HUGGINGFACE_API_KEY = None
            
            with patch('transformers.pipeline') as mock_pipeline:
                mock_client = Mock()
                mock_client.tokenizer.eos_token_id = 50256
                mock_pipeline.return_value = mock_client
                
                provider = HuggingFaceProvider(use_local=True)
                
                # Mock the pipeline response
                mock_result = [{"generated_text": "Test prompt Generated response"}]
                
                with patch('asyncio.to_thread', return_value=mock_result):
                    response = await provider.generate("Test prompt")
                    
                    assert isinstance(response, LLMResponse)
                    assert response.provider == LLMProvider.HUGGINGFACE


class TestLocalProvider:
    """Test Local (Ollama) provider."""
    
    def test_initialization(self):
        """Test Local provider initialization."""
        provider = LocalProvider()
        
        assert provider.model_name == "llama2"
        assert provider.provider_type == LLMProvider.LOCAL
        assert provider.base_url == "http://localhost:11434"
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test Local provider generation."""
        provider = LocalProvider()
        
        # Mock aiohttp response
        mock_response_data = {
            "response": "Test response",
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            response = await provider.generate("Test prompt")
            
            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.provider == LLMProvider.LOCAL
    
    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test getting available models from Ollama."""
        provider = LocalProvider()
        
        mock_models_data = {
            "models": [
                {"name": "llama2"},
                {"name": "codellama"},
                {"name": "mistral"}
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_models_data)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            models = await provider.get_available_models()
            
            assert isinstance(models, list)
            assert "llama2" in models
            assert "codellama" in models
            assert "mistral" in models
    
    @pytest.mark.asyncio
    async def test_validate_config(self):
        """Test Local provider config validation."""
        provider = LocalProvider()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            is_valid = await provider.validate_config()
            assert is_valid is True


class TestProviderFactory:
    """Test provider factory function."""
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            
            provider = create_provider(LLMProvider.OPENAI, "gpt-4", api_key="test-key")
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.model_name == "gpt-4"
    
    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        with patch('src.langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.ANTHROPIC_API_KEY = "test-key"
            
            provider = create_provider(LLMProvider.ANTHROPIC, "claude-3-opus-20240229", api_key="test-key")
            
            assert isinstance(provider, AnthropicProvider)
            assert provider.model_name == "claude-3-opus-20240229"
    
    def test_create_huggingface_provider(self):
        """Test creating HuggingFace provider."""
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            provider = create_provider(LLMProvider.HUGGINGFACE, "test-model", use_local=True)
            
            assert isinstance(provider, HuggingFaceProvider)
            assert provider.model_name == "test-model"
    
    def test_create_local_provider(self):
        """Test creating Local provider."""
        provider = create_provider(LLMProvider.LOCAL, "llama2")
        
        assert isinstance(provider, LocalProvider)
        assert provider.model_name == "llama2"
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider type."""
        with pytest.raises(ValueError):
            create_provider("unknown_provider", "model")


@pytest.fixture
def mock_settings():
    """Mock settings fixture."""
    with patch('src.langchain_rag_learning.core.config.get_settings') as mock:
        settings = Mock()
        settings.OPENAI_API_KEY = "test-openai-key"
        settings.ANTHROPIC_API_KEY = "test-anthropic-key"
        settings.HUGGINGFACE_API_KEY = "test-hf-key"
        mock.return_value = settings
        yield settings


class TestProviderIntegration:
    """Integration tests for providers."""
    
    @pytest.mark.asyncio
    async def test_provider_error_handling(self, mock_settings):
        """Test provider error handling."""
        provider = OpenAIProvider(api_key="test-key")
        
        # Mock client to raise exception
        mock_client = AsyncMock()
        mock_client.predict = AsyncMock(side_effect=Exception("Network error"))
        provider._client = mock_client
        
        with patch('asyncio.to_thread', side_effect=Exception("Network error")):
            with pytest.raises(LLMError) as exc_info:
                await provider.generate("Test prompt")
            
            assert "OpenAI generation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_provider_response_format(self, mock_settings):
        """Test provider response format consistency."""
        provider = OpenAIProvider(api_key="test-key")
        
        mock_client = AsyncMock()
        mock_client.predict = AsyncMock(return_value="Test response")
        provider._client = mock_client
        
        with patch('asyncio.to_thread', return_value="Test response"):
            response = await provider.generate("Test prompt", temperature=0.5, max_tokens=100)
            
            # Verify response structure
            assert hasattr(response, 'content')
            assert hasattr(response, 'model_name')
            assert hasattr(response, 'provider')
            assert hasattr(response, 'response_time')
            assert hasattr(response, 'metadata')
            
            # Verify metadata contains parameters
            assert response.metadata['temperature'] == 0.5
            assert response.metadata['max_tokens'] == 100