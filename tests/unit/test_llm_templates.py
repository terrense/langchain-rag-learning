"""Unit tests for LLM prompt templates."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.langchain_rag_learning.llm.templates import (
    PromptTemplateManager,
    PromptConfig,
    PromptType,
    Language,
    RAGQueryTemplate,
    RAGQueryWithHistoryTemplate,
    SummarizationTemplate,
    PromptBuilder,
    create_rag_prompt,
    create_summarization_prompt
)
from src.langchain_rag_learning.core.exceptions import ConfigurationError


class TestPromptConfig:
    """Test prompt configuration."""
    
    def test_prompt_config_creation(self):
        """Test creating prompt configuration."""
        config = PromptConfig(
            name="test_prompt",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.ENGLISH,
            template="Test template with {variable}",
            input_variables=["variable"],
            description="Test description"
        )
        
        assert config.name == "test_prompt"
        assert config.prompt_type == PromptType.RAG_QUERY
        assert config.language == Language.ENGLISH
        assert config.template == "Test template with {variable}"
        assert config.input_variables == ["variable"]
        assert config.description == "Test description"


class TestRAGQueryTemplate:
    """Test RAG query template."""
    
    def test_rag_template_creation(self):
        """Test creating RAG template."""
        config = PromptConfig(
            name="rag_test",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.ENGLISH,
            template="Context: {context}\nQuery: {query}\nAnswer:",
            input_variables=["context", "query"],
            description="Test RAG template"
        )
        
        template = RAGQueryTemplate(config)
        assert template.config == config
    
    def test_rag_template_formatting(self):
        """Test RAG template formatting."""
        config = PromptConfig(
            name="rag_test",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.ENGLISH,
            template="Context: {context}\nQuery: {query}\nAnswer:",
            input_variables=["context", "query"],
            description="Test RAG template"
        )
        
        template = RAGQueryTemplate(config)
        formatted = template.format(
            context="Test context information",
            query="What is the test about?"
        )
        
        expected = "Context: Test context information\nQuery: What is the test about?\nAnswer:"
        assert formatted == expected
    
    def test_rag_template_validation_error(self):
        """Test RAG template validation error."""
        config = PromptConfig(
            name="rag_test",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.ENGLISH,
            template="Context: {context}\nQuery: {query}\nAnswer:",
            input_variables=["context", "query"],
            description="Test RAG template"
        )
        
        template = RAGQueryTemplate(config)
        
        with pytest.raises(ValueError) as exc_info:
            template.format(context="Test context")  # Missing query
        
        assert "Missing required variables" in str(exc_info.value)


class TestRAGQueryWithHistoryTemplate:
    """Test RAG query with history template."""
    
    def test_rag_history_template_formatting(self):
        """Test RAG history template formatting."""
        config = PromptConfig(
            name="rag_history_test",
            prompt_type=PromptType.RAG_QUERY_WITH_HISTORY,
            language=Language.ENGLISH,
            template="History: {history}\nContext: {context}\nQuery: {query}\nAnswer:",
            input_variables=["history", "context", "query"],
            description="Test RAG history template"
        )
        
        template = RAGQueryWithHistoryTemplate(config)
        formatted = template.format(
            history="Previous conversation",
            context="Test context information",
            query="What is the test about?"
        )
        
        expected = "History: Previous conversation\nContext: Test context information\nQuery: What is the test about?\nAnswer:"
        assert formatted == expected


class TestSummarizationTemplate:
    """Test summarization template."""
    
    def test_summarization_template_formatting(self):
        """Test summarization template formatting."""
        config = PromptConfig(
            name="summary_test",
            prompt_type=PromptType.SUMMARIZATION,
            language=Language.ENGLISH,
            template="Summarize the following text:\n{text}\n\nSummary:",
            input_variables=["text"],
            description="Test summarization template"
        )
        
        template = SummarizationTemplate(config)
        formatted = template.format(text="This is a long text that needs to be summarized.")
        
        expected = "Summarize the following text:\nThis is a long text that needs to be summarized.\n\nSummary:"
        assert formatted == expected


class TestPromptTemplateManager:
    """Test prompt template manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization with default templates."""
        manager = PromptTemplateManager()
        
        # Check that default templates are loaded
        assert PromptType.RAG_QUERY in manager.templates
        assert Language.ENGLISH in manager.templates[PromptType.RAG_QUERY]
        assert Language.CHINESE in manager.templates[PromptType.RAG_QUERY]
    
    def test_get_template(self):
        """Test getting template."""
        manager = PromptTemplateManager()
        
        template = manager.get_template(PromptType.RAG_QUERY, Language.ENGLISH)
        assert isinstance(template, RAGQueryTemplate)
        
        template = manager.get_template(PromptType.RAG_QUERY_WITH_HISTORY, Language.ENGLISH)
        assert isinstance(template, RAGQueryWithHistoryTemplate)
    
    def test_get_template_fallback_to_english(self):
        """Test fallback to English when language not available."""
        manager = PromptTemplateManager()
        
        # Try to get a template in a language that doesn't exist
        template = manager.get_template(PromptType.RAG_QUERY, Language.SPANISH)
        assert isinstance(template, RAGQueryTemplate)
        # Should fallback to English
    
    def test_get_template_unknown_type(self):
        """Test getting unknown template type."""
        manager = PromptTemplateManager()
        
        with pytest.raises(ValueError):
            manager.get_template("unknown_type")
    
    def test_format_prompt(self):
        """Test formatting prompt through manager."""
        manager = PromptTemplateManager()
        
        formatted = manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context="Test context",
            query="Test query"
        )
        
        assert isinstance(formatted, str)
        assert "Test context" in formatted
        assert "Test query" in formatted
    
    def test_register_template(self):
        """Test registering new template."""
        manager = PromptTemplateManager()
        
        config = PromptConfig(
            name="custom_test",
            prompt_type=PromptType.EXPLANATION,
            language=Language.ENGLISH,
            template="Explain: {topic}",
            input_variables=["topic"],
            description="Custom explanation template"
        )
        
        manager.register_template(config)
        
        assert PromptType.EXPLANATION in manager.templates
        assert Language.ENGLISH in manager.templates[PromptType.EXPLANATION]
    
    def test_get_available_templates(self):
        """Test getting available templates."""
        manager = PromptTemplateManager()
        
        available = manager.get_available_templates()
        
        assert isinstance(available, dict)
        assert PromptType.RAG_QUERY in available
        assert Language.ENGLISH in available[PromptType.RAG_QUERY]
        assert Language.CHINESE in available[PromptType.RAG_QUERY]
    
    def test_validate_template(self):
        """Test template validation."""
        manager = PromptTemplateManager()
        
        # Valid inputs
        is_valid = manager.validate_template(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context="Test context",
            query="Test query"
        )
        assert is_valid is True
        
        # Invalid inputs (missing required variable)
        is_valid = manager.validate_template(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context="Test context"
            # Missing query
        )
        assert is_valid is False
    
    def test_set_default_language(self):
        """Test setting default language."""
        manager = PromptTemplateManager()
        
        manager.set_default_language(Language.CHINESE)
        assert manager.default_language == Language.CHINESE
    
    def test_get_template_info(self):
        """Test getting template information."""
        manager = PromptTemplateManager()
        
        info = manager.get_template_info(PromptType.RAG_QUERY, Language.ENGLISH)
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'prompt_type' in info
        assert 'language' in info
        assert 'input_variables' in info
        assert 'description' in info
    
    def test_create_chat_prompt(self):
        """Test creating chat prompt."""
        manager = PromptTemplateManager()
        
        chat_prompt = manager.create_chat_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            system_message="You are a helpful assistant."
        )
        
        # Should return a ChatPromptTemplate
        assert hasattr(chat_prompt, 'format_messages')


class TestPromptTemplateFileOperations:
    """Test file operations for templates."""
    
    def test_save_and_load_templates(self, tmp_path):
        """Test saving and loading templates from file."""
        manager = PromptTemplateManager()
        
        # Save templates to file
        file_path = tmp_path / "test_templates.json"
        manager.save_templates_to_file(file_path)
        
        assert file_path.exists()
        
        # Create new manager and load templates
        new_manager = PromptTemplateManager()
        new_manager.templates.clear()  # Clear default templates
        
        new_manager.load_templates_from_file(file_path)
        
        # Should have loaded templates
        assert len(new_manager.templates) > 0
    
    def test_load_templates_from_nonexistent_file(self):
        """Test loading templates from non-existent file."""
        manager = PromptTemplateManager()
        
        with pytest.raises(ConfigurationError):
            manager.load_templates_from_file("nonexistent.json")
    
    def test_load_templates_from_invalid_json(self, tmp_path):
        """Test loading templates from invalid JSON file."""
        manager = PromptTemplateManager()
        
        # Create invalid JSON file
        file_path = tmp_path / "invalid.json"
        file_path.write_text("invalid json content")
        
        with pytest.raises(ConfigurationError):
            manager.load_templates_from_file(file_path)


class TestPromptBuilder:
    """Test prompt builder."""
    
    def test_prompt_builder_basic(self):
        """Test basic prompt builder functionality."""
        manager = PromptTemplateManager()
        builder = PromptBuilder(manager)
        
        prompt = (builder
                 .add_system_message("You are a helpful assistant.")
                 .add_context("Test context information")
                 .add_template(PromptType.RAG_QUERY, Language.ENGLISH, query="What is this about?")
                 .build())
        
        assert isinstance(prompt, str)
        assert "System: You are a helpful assistant." in prompt
        assert "Test context information" in prompt
        assert "What is this about?" in prompt
    
    def test_prompt_builder_with_history(self):
        """Test prompt builder with conversation history."""
        manager = PromptTemplateManager()
        builder = PromptBuilder(manager)
        
        prompt = (builder
                 .add_history("Previous conversation")
                 .add_context("Test context")
                 .add_template(PromptType.RAG_QUERY_WITH_HISTORY, Language.ENGLISH, query="Follow-up question")
                 .build())
        
        assert "Previous conversation" in prompt
        assert "Test context" in prompt
        assert "Follow-up question" in prompt
    
    def test_prompt_builder_reset(self):
        """Test prompt builder reset."""
        manager = PromptTemplateManager()
        builder = PromptBuilder(manager)
        
        builder.add_context("Test context").add_variable("key", "value")
        
        # Reset and build new prompt
        prompt = (builder
                 .reset()
                 .add_template(PromptType.SUMMARIZATION, Language.ENGLISH, text="New text")
                 .build())
        
        assert "Test context" not in prompt
        assert "New text" in prompt


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_rag_prompt_without_history(self):
        """Test creating RAG prompt without history."""
        prompt = create_rag_prompt(
            query="What is machine learning?",
            context="Machine learning is a subset of AI...",
            language=Language.ENGLISH
        )
        
        assert isinstance(prompt, str)
        assert "What is machine learning?" in prompt
        assert "Machine learning is a subset of AI..." in prompt
    
    def test_create_rag_prompt_with_history(self):
        """Test creating RAG prompt with history."""
        prompt = create_rag_prompt(
            query="Can you elaborate?",
            context="Machine learning is a subset of AI...",
            history="User: What is AI? Assistant: AI is artificial intelligence.",
            language=Language.ENGLISH
        )
        
        assert isinstance(prompt, str)
        assert "Can you elaborate?" in prompt
        assert "Machine learning is a subset of AI..." in prompt
        assert "What is AI?" in prompt
    
    def test_create_summarization_prompt(self):
        """Test creating summarization prompt."""
        prompt = create_summarization_prompt(
            text="This is a long document that needs to be summarized...",
            language=Language.ENGLISH
        )
        
        assert isinstance(prompt, str)
        assert "This is a long document that needs to be summarized..." in prompt


class TestMultiLanguageSupport:
    """Test multi-language support."""
    
    def test_chinese_templates(self):
        """Test Chinese template formatting."""
        manager = PromptTemplateManager()
        
        formatted = manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.CHINESE,
            context="测试上下文",
            query="这是什么？"
        )
        
        assert isinstance(formatted, str)
        assert "测试上下文" in formatted
        assert "这是什么？" in formatted
    
    def test_language_fallback(self):
        """Test language fallback mechanism."""
        manager = PromptTemplateManager()
        
        # Try to get template in unsupported language
        template = manager.get_template(PromptType.RAG_QUERY, Language.GERMAN)
        
        # Should fallback to English
        assert isinstance(template, RAGQueryTemplate)


class TestPromptTemplateIntegration:
    """Integration tests for prompt templates."""
    
    def test_template_with_real_data(self):
        """Test template with realistic data."""
        manager = PromptTemplateManager()
        
        context = """
        LangChain is a framework for developing applications powered by language models.
        It provides tools for prompt management, chains, and agents.
        The framework supports multiple LLM providers including OpenAI and Anthropic.
        """
        
        query = "What is LangChain and what does it provide?"
        
        formatted = manager.format_prompt(
            PromptType.RAG_QUERY,
            Language.ENGLISH,
            context=context.strip(),
            query=query
        )
        
        assert "LangChain is a framework" in formatted
        assert "What is LangChain" in formatted
        assert "Context:" in formatted or "context" in formatted.lower()
    
    def test_conversation_flow(self):
        """Test conversation flow with history."""
        manager = PromptTemplateManager()
        
        history = "User: What is machine learning? Assistant: Machine learning is a method of data analysis."
        context = "Deep learning is a subset of machine learning that uses neural networks."
        query = "How does deep learning relate to what we discussed?"
        
        formatted = manager.format_prompt(
            PromptType.RAG_QUERY_WITH_HISTORY,
            Language.ENGLISH,
            history=history,
            context=context,
            query=query
        )
        
        assert "machine learning" in formatted
        assert "deep learning" in formatted
        assert "neural networks" in formatted
        assert "How does deep learning relate" in formatted