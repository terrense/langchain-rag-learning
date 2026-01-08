"""
Configuration management for the LangChain RAG Learning project.

This module provides a flexible configuration system that supports:
- Environment variable loading via .env files
- YAML-based LLM provider configuration  
- Dynamic provider switching and management
- Validation and error handling

Technical Features:
- Pydantic for data validation and settings management
- YAML parsing for complex configuration structures
- Environment variable substitution in YAML files
- Caching for performance optimization
"""

import os  # Operating system interface

# Try to import functools.lru_cache for performance optimization
# lru_cache provides memoization (caching) for function results
try:
    from functools import lru_cache
    FUNCTOOLS_AVAILABLE = True
except ImportError:
    # Fallback decorator if functools is not available
    FUNCTOOLS_AVAILABLE = False
    def lru_cache():
        """Fallback decorator that does nothing if lru_cache is unavailable."""
        def decorator(func):
            """
            Decorator function implementation.
            """
            return func
        return decorator
from pathlib import Path  # Modern cross-platform path handling  # Modern cross-platform path handling
from typing import Optional, List, Dict, Any, Union  # Type hints for better code documentation  # Type hints for better code documentation

# Try to import Pydantic for data validation and settings management
# Pydantic provides automatic validation, serialization, and documentation
try:
    from pydantic_settings import BaseSettings  # Modern Pydantic settings import  # Data validation and serialization
    from pydantic import Field, validator  # Field definitions and custom validators  # Data validation and serialization
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback to older Pydantic import structure
    try:
        from pydantic import BaseSettings, Field, validator  # Data validation and serialization
        PYDANTIC_AVAILABLE = True
    except ImportError:
        # Complete fallback for when Pydantic is not available
        PYDANTIC_AVAILABLE = False
        
        class BaseSettings:
            """Fallback BaseSettings class when Pydantic is unavailable."""
            def __init__(self, **kwargs):
                """
                  Init   function implementation.
                """
                # Simple attribute assignment for basic functionality
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        def Field(default=None, env=None, **kwargs):
            """Fallback Field function that just returns the default value."""
            return default
        
        def validator(field_name, **kwargs):
            """Fallback validator decorator that does nothing."""
            def decorator(func):
                """
                Decorator function implementation.
                """
                return func
            return decorator

# Try to import YAML parser for configuration files
# YAML provides human-readable configuration format
try:
    import yaml  # YAML parsing for configuration files
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Simple logger for configuration warnings and debugging
import logging  # Structured logging for debugging and monitoring
logger = logging.getLogger(__name__)


class LLMProviderConfig:
    """
    Configuration manager for LLM providers loaded from YAML files.
    
    This class handles:
    - Loading provider configurations from YAML
    - Environment variable substitution (${VAR_NAME} syntax)
    - Provider availability checking
    - Default configuration creation
    
    Design Pattern: Singleton-like behavior through module-level caching
    """
    
    def __init__(self, config_path: str = "config/llm_providers.yaml"):
        """
        Initialize the LLM provider configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Technical Notes:
        - Uses pathlib.Path for cross-platform path handling
        - Implements fallback configuration if file doesn't exist
        """
        self.config_path = Path(config_path)  # Convert string to Path object for better handling
        self._config = None  # Private attribute to store loaded configuration
        self.load_config()  # Load configuration immediately upon initialization
    
    def load_config(self):
        """
        Load configuration from YAML file with comprehensive error handling.
        
        Process Flow:
        1. Check if config file exists on filesystem
        2. Parse YAML content safely (prevents code execution)
        3. Replace environment variables using ${VAR} syntax
        4. Create default config if file doesn't exist
        
        Error Handling:
        - Graceful fallback to default configuration on any error
        - Warning messages for debugging and troubleshooting
        """
        try:
            if self.config_path.exists():  # Check if file exists before attempting to read
                if not YAML_AVAILABLE:
                    logger.warning("PyYAML not available, using default configuration")
                    self._config = self._get_default_config()
                    return
                    
                # Open file with UTF-8 encoding for international character support
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    # yaml.safe_load prevents arbitrary code execution (security feature)
                    self._config = yaml.safe_load(f)
                    # Replace environment variables in the loaded configuration
                    self._config = self._replace_env_vars(self._config)
            else:
                # Auto-create default configuration for first-time setup
                self._create_default_config()
        except Exception as e:
            # Catch all exceptions to ensure system continues working
            print(f"Warning: Could not load LLM config from {self.config_path}: {e}")
            # Fallback to hardcoded default configuration
            self._config = self._get_default_config()
    
    def _replace_env_vars(self, obj):
        """
        Recursively replace environment variables in configuration data structures.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        This allows configuration files to reference environment variables securely.
        
        Args:
            obj: Configuration object (can be dict, list, string, or other types)
            
        Returns:
            Object with environment variables replaced by their actual values
            
        Technical Implementation:
        - Uses recursive traversal for nested data structures
        - Pattern matching for ${VAR} syntax using string methods
        - Safe fallback to empty string if environment variable doesn't exist
        - Preserves original data types and structure
        """
        if isinstance(obj, dict):
            # Recursively process all dictionary key-value pairs
            # Dictionary comprehension: {key: processed_value for key, value in items}
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process all list items
            # List comprehension: [processed_item for item in list]
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Extract environment variable name from ${VAR_NAME} format
            env_var = obj[2:-1]  # String slicing: remove first 2 chars (${) and last 1 char (})
            # os.getenv() safely retrieves environment variable with fallback to empty string
            return os.getenv(env_var, "")
        # Return unchanged for all other data types (int, float, bool, etc.)
        return obj
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            "default_provider": "deepseek",
            "providers": {
                "deepseek": {
                    "type": "openai_compatible",
                    "name": "DeepSeek",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                    "models": ["deepseek-chat", "deepseek-coder"],
                    "default_model": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "enabled": True
                },
                "ollama": {
                    "type": "ollama",
                    "name": "Ollama Local",
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "models": ["llama2", "codellama", "mistral"],
                    "default_model": "llama2",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "enabled": True
                }
            },
            "embeddings": {
                "default_provider": "sentence_transformers",
                "providers": {
                    "sentence_transformers": {
                        "type": "sentence_transformers",
                        "model": "all-MiniLM-L6-v2",
                        "enabled": True
                    }
                }
            }
        }
    
    def _create_default_config(self):
        """Create default config file."""
        self._config = self._get_default_config()
        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default config
        if YAML_AVAILABLE:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        else:
            # Write as JSON if YAML not available
            import json  # JSON parsing and serialization
            with open(self.config_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self._config.get("default_provider", "deepseek")
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        providers = self._config.get("providers", {})
        if provider_name not in providers:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")
        return providers[provider_name]
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers."""
        providers = self._config.get("providers", {})
        return [name for name, config in providers.items() if config.get("enabled", False)]
    
    def get_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get all provider configurations."""
        return self._config.get("providers", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self._config.get("embeddings", {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self._config.get("rag", {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
            "rerank_top_k": 3,
            "similarity_threshold": 0.7
        })
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (has API key if required)."""
        try:
            config = self.get_provider_config(provider_name)
            if not config.get("enabled", False):
                return False
            
            # Check if API key is required and available
            api_key = config.get("api_key", "")
            provider_type = config.get("type", "")
            
            # Local providers don't need API keys
            if provider_type in ["ollama", "sentence_transformers"]:
                return True
            
            # Other providers need API keys
            return bool(api_key and api_key.strip())
        except Exception:
            return False


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    def __init__(self, **kwargs):
        """
          Init   function implementation.
        """
        if PYDANTIC_AVAILABLE:
            super().__init__(**kwargs)
        else:
            # Fallback initialization
            self._init_fallback_settings()
    
    def _init_fallback_settings(self):
        """Initialize settings without pydantic."""
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
        
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # LLM API Keys
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
        self.ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
        self.MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
        
        # LLM Cache Settings
        self.llm_cache_ttl = int(os.getenv("LLM_CACHE_TTL", "3600"))
        self.enable_llm_cache = os.getenv("ENABLE_LLM_CACHE", "True").lower() == "true"
        
        # LLM Provider Settings
        self.default_llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.fallback_providers = os.getenv("FALLBACK_PROVIDERS", "openai,anthropic").split(",")
        
        # Rate Limiting
        self.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "False").lower() == "true"
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
        self.enable_model_fallback = os.getenv("ENABLE_MODEL_FALLBACK", "True").lower() == "true"
        
        # Vector Database Configuration
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
        self.CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
        self.CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
        
        # File Upload Configuration
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
        self.ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", ".pdf,.txt,.md,.docx,.html").split(",")
        
        # Security Configuration
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE")
        
        # Development Configuration
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider."""
        configs = {
            'openai': {
                'api_key': self.OPENAI_API_KEY,
            },
            'anthropic': {
                'api_key': self.ANTHROPIC_API_KEY,
            },
            'huggingface': {
                'api_key': self.HUGGINGFACE_API_KEY,
            },
            'local': {
                'base_url': 'http://localhost:11434',
            }
        }
        
        config = configs.get(provider, {})
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
    
    # Pydantic field definitions (only used when pydantic is available)
    if PYDANTIC_AVAILABLE:
        # API Configuration
        API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
        API_PORT: int = Field(default=8000, env="API_PORT")
        API_RELOAD: bool = Field(default=True, env="API_RELOAD")
        
        # Database Configuration
        DATABASE_URL: str = Field(default="sqlite:///./data/app.db", env="DATABASE_URL")
        REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
        
        # LLM API Keys (now loaded from config file)
        DEEPSEEK_API_KEY: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
        OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
        ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
        HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
        ZHIPU_API_KEY: Optional[str] = Field(default=None, env="ZHIPU_API_KEY")
        MOONSHOT_API_KEY: Optional[str] = Field(default=None, env="MOONSHOT_API_KEY")
        
        # Vector Database Configuration
        CHROMA_PERSIST_DIRECTORY: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIRECTORY")
        CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
        CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")
        
        # File Upload Configuration
        MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
        UPLOAD_DIR: str = Field(default="./data/uploads", env="UPLOAD_DIR")
        ALLOWED_FILE_TYPES: List[str] = Field(
            default=[".pdf", ".txt", ".md", ".docx", ".html"],
            env="ALLOWED_FILE_TYPES"
        )
        
        # Security Configuration
        SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
        JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
        JWT_EXPIRE_MINUTES: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
        
        # Logging Configuration
        LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
        LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
        
        # Development Configuration
        DEBUG: bool = Field(default=False, env="DEBUG")
        ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        if not PYDANTIC_AVAILABLE:
            return v
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("ALLOWED_FILE_TYPES")
    def validate_file_types(cls, v):
        """Ensure file types start with a dot."""
        if not PYDANTIC_AVAILABLE:
            return v
        if isinstance(v, str):
            v = [ext.strip() for ext in v.split(",")]
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]
    
    if PYDANTIC_AVAILABLE:
        class Config:
            """Pydantic configuration."""
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            extra = "ignore"  # Allow extra fields


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache()
def get_llm_config() -> LLMProviderConfig:
    """Get cached LLM configuration instance."""
    return LLMProviderConfig()


# Global settings instance for backward compatibility
settings = get_settings()


def load_config_from_file(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")
    
    try:
        if config_path.suffix.lower() == '.json':
            import json  # JSON parsing and serialization
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            if not YAML_AVAILABLE:
                raise ValueError("PyYAML is required to load YAML configuration files")
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    except Exception as e:
        raise ValueError(f"Error loading configuration file {config_path}: {e}")


def update_settings_from_dict(config_dict: Dict[str, Any]) -> None:
    """Update global settings from dictionary."""
    global settings
    
    # Convert dictionary keys to uppercase for environment variable style
    env_dict = {}
    for key, value in config_dict.items():
        env_key = key.upper()
        if isinstance(value, list):
            env_dict[env_key] = ','.join(str(v) for v in value)
        else:
            env_dict[env_key] = str(value)
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in env_dict.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Reload settings
        settings = Settings()
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def validate_environment() -> None:
    """Validate the current environment configuration."""
    # Basic validation - can be expanded
    if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-change-in-production":
        logger.warning("Using default SECRET_KEY - change this in production!")


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration (excluding sensitive data)."""
    if PYDANTIC_AVAILABLE:
        config = settings.dict() if hasattr(settings, 'dict') else settings.__dict__
    else:
        config = settings.__dict__
    
    # Mask sensitive information
    sensitive_keys = [
        'SECRET_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 
        'ANTHROPIC_API_KEY', 'HUGGINGFACE_API_KEY', 'DATABASE_URL'
    ]
    
    for key in sensitive_keys:
        if key in config and config[key]:
            config[key] = f"{'*' * (len(str(config[key])) - 4)}{str(config[key])[-4:]}"
    
    return config


class ConfigManager:
    """Configuration manager for dynamic configuration updates."""
    
    def __init__(self):
        """
          Init   function implementation.
        """
        self._settings = settings
        self._config_cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self._settings, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (runtime only)."""
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
    
    def reload_from_env(self) -> None:
        """Reload configuration from environment variables."""
        global settings
        settings = Settings()
        self._settings = settings
        self.clear_cache()


# Global config manager instance
config_manager = ConfigManager()