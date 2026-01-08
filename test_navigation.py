"""Test file to verify IDE navigation functionality."""

from src.langchain_rag_learning.core.config import get_settings, get_llm_config
from src.langchain_rag_learning.core.models import LLMProvider
from src.langchain_rag_learning.llm.providers import BaseLLMProvider

def test_navigation():
    """Test function to verify Ctrl+Click navigation works."""
    
    # Try to get settings - Ctrl+Click on get_settings should jump to definition
    settings = get_settings()
    
    # Try to get LLM config - Ctrl+Click on get_llm_config should jump to definition
    llm_config = get_llm_config()
    
    # Try enum - Ctrl+Click on LLMProvider should jump to definition
    provider_type = LLMProvider.DEEPSEEK
    
    # Try class - Ctrl+Click on BaseLLMProvider should jump to definition
    print(f"BaseLLMProvider: {BaseLLMProvider}")
    
    print("Navigation test completed!")

if __name__ == "__main__":
    test_navigation()