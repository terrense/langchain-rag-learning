"""
LLM provider implementations for different AI services.

This module implements the provider pattern to support multiple LLM services:
- OpenAI GPT models (GPT-3.5, GPT-4)
- Anthropic Claude models
- HuggingFace transformers (local and API)
- Local models via Ollama
- Custom OpenAI-compatible APIs (DeepSeek, etc.)

Architecture:
- Abstract base class defines common interface
- Concrete implementations for each provider
- Factory pattern for provider instantiation
- Async/await for non-blocking I/O operations
- Error handling and retry mechanisms
"""

import asyncio  # Async programming support for concurrent operations  # Async programming support for concurrent operations
import json    # JSON parsing for API responses  # JSON parsing and serialization
import time    # Time utilities for performance measurement  # Time utilities for performance measurement
from abc import ABC, abstractmethod  # Abstract base class support
from typing import Any, Dict, List, Optional, Union  # Type hints for better code documentation  # Type hints for better code documentation

# Try to import aiohttp for async HTTP requests
# aiohttp provides non-blocking HTTP client functionality
try:
    import aiohttp  # Async HTTP client for non-blocking requests
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Try to import LangChain components for LLM integration
# LangChain provides unified interfaces for different LLM providers
try:
    from langchain.llms import OpenAI  # OpenAI LLM wrapper  # LangChain framework for LLM applications
    from langchain.chat_models import ChatOpenAI, ChatAnthropic  # Chat model wrappers  # LangChain framework for LLM applications
    from langchain.llms import HuggingFacePipeline  # Local HuggingFace model support  # LangChain framework for LLM applications
    from langchain.callbacks.manager import CallbackManagerForLLMRun  # Callback system  # LangChain framework for LLM applications
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Try to import HuggingFace transformers for local model execution
# Transformers library provides pre-trained models and tokenizers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # HuggingFace transformers for NLP models
    import torch  # PyTorch for deep learning operations  # PyTorch for deep learning
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import project-specific modules
from ..core.config import get_settings  # Configuration management  # Regular expressions for text processing
from ..core.exceptions import LLMError, ConfigurationError  # Custom exception classes  # Regular expressions for text processing
from ..core.logging import get_logger  # Structured logging  # Structured logging for debugging and monitoring
from ..core.models import LLMProvider, LLMResponse  # Data models  # Regular expressions for text processing

# Initialize logger for this module with structured logging
logger = get_logger(__name__)
# Get application settings (cached for performance via lru_cache)
settings = get_settings()


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the common interface that all LLM providers must implement.
    It follows the Template Method pattern, providing a consistent API while
    allowing each provider to implement its specific functionality.
    
    Design Patterns Used:
    - Abstract Base Class: Enforces interface compliance
    - Template Method: Common initialization and error handling
    - Strategy Pattern: Different providers implement same interface
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            model_name: Name of the specific model to use (e.g., "gpt-3.5-turbo")
            **kwargs: Additional configuration parameters specific to each provider
            
        Instance Variables:
        - model_name: The specific model identifier
        - config: Dictionary of provider-specific configuration
        - provider_type: Enum identifying the provider type
        - _client: Private attribute for the underlying client connection
        """
        self.model_name = model_name  # Store the model identifier
        self.config = kwargs  # Store all additional configuration parameters
        self.provider_type = None  # Will be set by concrete implementations
        self._client = None  # Private client instance (lazy initialization)
        
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass
    
    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate provider configuration."""
        pass
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            await self.validate_config()
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_type}: {e}")
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        """
          Init   function implementation.
        """
        super().__init__(model_name, **kwargs)
        self.provider_type = LLMProvider.OPENAI
        self.api_key = kwargs.get('api_key') or settings.OPENAI_API_KEY
        self.base_url = kwargs.get('base_url', "https://api.openai.com/v1")
        self.organization = kwargs.get('organization')
        
        if not LANGCHAIN_AVAILABLE:
            raise ConfigurationError("LangChain is required for OpenAI provider")
            
        if not self.api_key:
            raise ConfigurationError("OpenAI API key is required")
            
        # Initialize LangChain client
        self._client = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            openai_organization=self.organization,
            openai_api_base=self.base_url,
            **kwargs
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            # Set parameters
            if max_tokens:
                self._client.max_tokens = max_tokens
            if temperature is not None:
                self._client.temperature = temperature
                
            # Generate response
            response = await asyncio.to_thread(self._client.predict, prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response,
                model_name=self.model_name,
                provider=self.provider_type,
                usage=kwargs.get('usage', {}),
                response_time=response_time,
                metadata={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    **kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMError(f"OpenAI generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response using OpenAI API."""
        try:
            # Set parameters
            if max_tokens:
                self._client.max_tokens = max_tokens
            if temperature is not None:
                self._client.temperature = temperature
                
            # Enable streaming
            self._client.streaming = True
            
            # Generate streaming response
            for chunk in self._client.stream(prompt):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
                
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise LLMError(f"OpenAI streaming failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "text-davinci-003",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001"
        ]
    
    async def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        if not self.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        try:
            # Test with a simple request
            test_response = await self.generate("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"OpenAI config validation failed: {e}")
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider implementation."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        """
          Init   function implementation.
        """
        super().__init__(model_name, **kwargs)
        self.provider_type = LLMProvider.ANTHROPIC
        self.api_key = kwargs.get('api_key') or settings.ANTHROPIC_API_KEY
        
        if not LANGCHAIN_AVAILABLE:
            raise ConfigurationError("LangChain is required for Anthropic provider")
            
        if not self.api_key:
            raise ConfigurationError("Anthropic API key is required")
            
        # Initialize LangChain client
        self._client = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            **kwargs
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            # Set parameters
            if max_tokens:
                self._client.max_tokens = max_tokens
            if temperature is not None:
                self._client.temperature = temperature
                
            # Generate response
            response = await asyncio.to_thread(self._client.predict, prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response,
                model_name=self.model_name,
                provider=self.provider_type,
                usage=kwargs.get('usage', {}),
                response_time=response_time,
                metadata={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    **kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise LLMError(f"Anthropic generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response using Anthropic API."""
        try:
            # Set parameters
            if max_tokens:
                self._client.max_tokens = max_tokens
            if temperature is not None:
                self._client.temperature = temperature
                
            # Enable streaming
            self._client.streaming = True
            
            # Generate streaming response
            for chunk in self._client.stream(prompt):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
                
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise LLMError(f"Anthropic streaming failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    async def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        if not self.api_key:
            raise ConfigurationError("Anthropic API key is required")
        
        try:
            # Test with a simple request
            test_response = await self.generate("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Anthropic config validation failed: {e}")
            return False


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace transformers LLM provider implementation."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        """
          Init   function implementation.
        """
        super().__init__(model_name, **kwargs)
        self.provider_type = LLMProvider.HUGGINGFACE
        self.api_key = kwargs.get('api_key') or settings.HUGGINGFACE_API_KEY
        self.use_local = kwargs.get('use_local', True)
        self.device = kwargs.get('device', 'auto')
        
        if self.use_local:
            self._init_local_model()
        else:
            self._init_api_client()
    
    def _init_local_model(self):
        """Initialize local HuggingFace model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError("Transformers library is required for local HuggingFace models")
            
        try:
            # Determine device
            if self.device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = self.device
                
            # Initialize pipeline
            self._client = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **self.config
            )
            
            logger.info(f"Initialized local HuggingFace model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize local HuggingFace model: {e}")
            raise ConfigurationError(f"Failed to initialize HuggingFace model: {str(e)}")
    
    def _init_api_client(self):
        """Initialize HuggingFace API client."""
        if not self.api_key:
            raise ConfigurationError("HuggingFace API key is required for API usage")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using HuggingFace."""
        start_time = time.time()
        
        try:
            if self.use_local:
                response = await self._generate_local(prompt, max_tokens, temperature, **kwargs)
            else:
                response = await self._generate_api(prompt, max_tokens, temperature, **kwargs)
                
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response,
                model_name=self.model_name,
                provider=self.provider_type,
                usage=kwargs.get('usage', {}),
                response_time=response_time,
                metadata={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'use_local': self.use_local,
                    **kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise LLMError(f"HuggingFace generation failed: {str(e)}")
    
    async def _generate_local(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate using local model."""
        generation_kwargs = {
            'max_new_tokens': max_tokens or 100,
            'temperature': temperature or 0.7,
            'do_sample': True,
            'pad_token_id': self._client.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate response in thread to avoid blocking
        result = await asyncio.to_thread(
            self._client,
            prompt,
            **generation_kwargs
        )
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        
        return ""
    
    async def _generate_api(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate using HuggingFace API."""
        if not AIOHTTP_AVAILABLE:
            raise ConfigurationError("aiohttp is required for HuggingFace API usage")
            
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or 100,
                "temperature": temperature or 0.7,
                "return_full_text": False,
                **kwargs
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"HuggingFace API error: {error_text}")
                
                result = await response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                
                return ""
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response (not supported for HuggingFace)."""
        # HuggingFace doesn't support streaming in the same way
        # We'll simulate it by generating the full response and yielding it
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response.content
    
    async def get_available_models(self) -> List[str]:
        """Get popular HuggingFace models."""
        return [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1"
        ]
    
    async def validate_config(self) -> bool:
        """Validate HuggingFace configuration."""
        try:
            if self.use_local:
                # Test local model
                if self._client is None:
                    return False
                test_response = await self._generate_local("Hello", max_tokens=1)
                return True
            else:
                # Test API
                if not self.api_key:
                    return False
                test_response = await self._generate_api("Hello", max_tokens=1)
                return True
        except Exception as e:
            logger.error(f"HuggingFace config validation failed: {e}")
            return False


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider implementation."""
    
    def __init__(self, model_name: str = "deepseek-chat", **kwargs):
        """
          Init   function implementation.
        """
        super().__init__(model_name, **kwargs)
        self.provider_type = LLMProvider.DEEPSEEK
        self.api_key = kwargs.get('api_key') or settings.DEEPSEEK_API_KEY
        self.base_url = kwargs.get('base_url', "https://api.deepseek.com/v1")
        
        if not self.api_key:
            raise ConfigurationError("DeepSeek API key is required")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using DeepSeek API."""
        if not AIOHTTP_AVAILABLE:
            raise ConfigurationError("aiohttp is required for DeepSeek API usage")
            
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or 1000,
                "temperature": temperature or 0.7,
                "stream": False,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"DeepSeek API error: {error_text}")
                    
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    content = result['choices'][0]['message']['content']
                    usage = result.get('usage', {})
                    
                    return LLMResponse(
                        content=content,
                        model_name=self.model_name,
                        provider=self.provider_type,
                        usage=usage,
                        response_time=response_time,
                        metadata={
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            **kwargs
                        }
                    )
                    
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise LLMError(f"DeepSeek generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response using DeepSeek API."""
        if not AIOHTTP_AVAILABLE:
            raise ConfigurationError("aiohttp is required for DeepSeek API streaming")
            
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or 1000,
                "temperature": temperature or 0.7,
                "stream": True,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"DeepSeek streaming error: {error_text}")
                    
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue
                                    
        except Exception as e:
            logger.error(f"DeepSeek streaming failed: {e}")
            raise LLMError(f"DeepSeek streaming failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        """Get available DeepSeek models."""
        return [
            "deepseek-chat",
            "deepseek-coder"
        ]
    
    async def validate_config(self) -> bool:
        """Validate DeepSeek configuration."""
        if not self.api_key:
            raise ConfigurationError("DeepSeek API key is required")
        
        try:
            # Test with a simple request
            test_response = await self.generate("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"DeepSeek config validation failed: {e}")
            return False


class LocalProvider(BaseLLMProvider):
    """Local model provider using Ollama or similar local inference servers."""
    
    def __init__(self, model_name: str = "llama2", **kwargs):
        """
          Init   function implementation.
        """
        super().__init__(model_name, **kwargs)
        self.provider_type = LLMProvider.LOCAL
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        self.timeout = kwargs.get('timeout', 30)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local Ollama server."""
        if not AIOHTTP_AVAILABLE:
            raise ConfigurationError("aiohttp is required for local model communication")
            
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens or 100,
                    "temperature": temperature or 0.7,
                    **kwargs
                }
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Local model error: {error_text}")
                    
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    return LLMResponse(
                        content=result.get('response', ''),
                        model_name=self.model_name,
                        provider=self.provider_type,
                        usage={
                            'prompt_tokens': result.get('prompt_eval_count', 0),
                            'completion_tokens': result.get('eval_count', 0),
                            'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                        },
                        response_time=response_time,
                        metadata={
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            'eval_duration': result.get('eval_duration', 0),
                            **kwargs
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise LLMError(f"Local model generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Generate streaming response using local Ollama server."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_tokens or 100,
                    "temperature": temperature or 0.7,
                    **kwargs
                }
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Local model streaming error: {error_text}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Local model streaming failed: {e}")
            raise LLMError(f"Local model streaming failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        """Get available models from Ollama server."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        return [model['name'] for model in result.get('models', [])]
                    else:
                        logger.warning("Could not fetch models from Ollama server")
                        return ["llama2", "codellama", "mistral", "neural-chat"]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return ["llama2", "codellama", "mistral", "neural-chat"]
    
    async def validate_config(self) -> bool:
        """Validate local server configuration."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Local server validation failed: {e}")
            return False


# Provider factory function
def create_provider(provider_type: str, model_name: str, **kwargs) -> BaseLLMProvider:
    """Factory function to create LLM providers."""
    providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.HUGGINGFACE: HuggingFaceProvider,
        LLMProvider.LOCAL: LocalProvider,
        LLMProvider.DEEPSEEK: DeepSeekProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return providers[provider_type](model_name, **kwargs)