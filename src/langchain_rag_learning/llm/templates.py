"""Prompt engineering templates for RAG queries and other LLM tasks."""

import json  # JSON parsing and serialization
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union  # Type hints for better code documentation
from enum import Enum
from dataclasses import dataclass
from pathlib import Path  # Modern cross-platform path handling

try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  # LangChain framework for LLM applications
    from langchain.schema import BaseMessage  # LangChain framework for LLM applications
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
    # Fallback implementations
    class PromptTemplate:
        """
        PromptTemplate class implementation.
        """
        def __init__(self, template: str, input_variables: List[str]):
            """
              Init   function implementation.
            """
            self.template = template
            self.input_variables = input_variables
        
        def format(self, **kwargs) -> str:
            """
            Format function implementation.
            """
            return self.template.format(**kwargs)
    
    class ChatPromptTemplate:
        """
        ChatPromptTemplate class implementation.
        """
        @classmethod
        def from_messages(cls, messages):
            """
            From Messages function implementation.
            """
            return cls()
    
    class SystemMessagePromptTemplate:
        """
        SystemMessagePromptTemplate class implementation.
        """
        @classmethod
        def from_template(cls, template):
            """
            From Template function implementation.
            """
            return cls()
    
    class HumanMessagePromptTemplate:
        """
        HumanMessagePromptTemplate class implementation.
        """
        @classmethod
        def from_template(cls, template):
            """
            From Template function implementation.
            """
            return cls()
    
    class BaseMessage:
        """
        BaseMessage class implementation.
        """
        pass

from ..core.config import get_settings  # Regular expressions for text processing
from ..core.logging import get_logger  # Structured logging for debugging and monitoring
from ..core.exceptions import ConfigurationError  # Regular expressions for text processing

logger = get_logger(__name__)
settings = get_settings()


class PromptType(str, Enum):
    """Types of prompts."""
    RAG_QUERY = "rag_query"
    RAG_QUERY_WITH_HISTORY = "rag_query_with_history"
    SUMMARIZATION = "summarization"
    QUESTION_GENERATION = "question_generation"
    CONTEXT_COMPRESSION = "context_compression"
    QUERY_REWRITING = "query_rewriting"
    FACT_CHECKING = "fact_checking"
    EXPLANATION = "explanation"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"


@dataclass
class PromptConfig:
    """Configuration for a prompt template."""
    name: str
    prompt_type: PromptType
    language: Language
    template: str
    input_variables: List[str]
    description: str
    examples: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, Any]] = None


class BasePromptTemplate(ABC):
    """Base class for prompt templates."""
    
    def __init__(self, config: PromptConfig):
        """
          Init   function implementation.
        """
        self.config = config
        self.template = PromptTemplate(
            template=config.template,
            input_variables=config.input_variables
        )
    
    def format(self, **kwargs) -> str:
        """Format the prompt with given variables."""
        self.validate_inputs(**kwargs)
        return self.template.format(**kwargs)
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate that all required inputs are provided."""
        missing_vars = set(self.config.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        return True


class RAGQueryTemplate(BasePromptTemplate):
    """Template for RAG query prompts."""
    
    def format(self, query: str, context: str, **kwargs) -> str:
        """Format RAG query prompt."""
        self.validate_inputs(query=query, context=context, **kwargs)
        return self.template.format(query=query, context=context, **kwargs)


class RAGQueryWithHistoryTemplate(BasePromptTemplate):
    """Template for RAG query prompts with conversation history."""
    
    def format(self, query: str, context: str, history: str, **kwargs) -> str:
        """Format RAG query prompt with history."""
        self.validate_inputs(query=query, context=context, history=history, **kwargs)
        return self.template.format(query=query, context=context, history=history, **kwargs)


class SummarizationTemplate(BasePromptTemplate):
    """Template for summarization prompts."""
    
    def format(self, text: str, **kwargs) -> str:
        """Format summarization prompt."""
        self.validate_inputs(text=text, **kwargs)
        return self.template.format(text=text, **kwargs)


class PromptTemplateManager:
    """Manager for prompt templates with multi-language support."""
    
    def __init__(self):
        """
          Init   function implementation.
        """
        self.templates: Dict[str, Dict[str, BasePromptTemplate]] = {}
        self.default_language = Language.ENGLISH
        
        # Load default templates
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # English RAG Query Template
        rag_query_en = PromptConfig(
            name="rag_query",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.ENGLISH,
            template="""You are a helpful AI assistant. Use the following context to answer the user's question. If you cannot answer the question based on the context provided, say so clearly.

Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, state this clearly
- Include relevant details from the context to support your answer
- Be concise but comprehensive
- If asked about sources, refer to the provided context

Answer:""",
            input_variables=["query", "context"],
            description="Standard RAG query template for answering questions based on retrieved context"
        )
        
        # Chinese RAG Query Template
        rag_query_zh = PromptConfig(
            name="rag_query",
            prompt_type=PromptType.RAG_QUERY,
            language=Language.CHINESE,
            template="""你是一个有用的AI助手。请根据以下上下文回答用户的问题。如果你无法基于提供的上下文回答问题，请明确说明。

上下文：
{context}

问题：{query}

指示：
- 基于上下文提供清晰、准确的答案
- 如果上下文不包含足够的信息，请明确说明
- 包含上下文中的相关细节来支持你的答案
- 简洁但全面
- 如果被问及来源，请参考提供的上下文

答案：""",
            input_variables=["query", "context"],
            description="标准RAG查询模板，用于基于检索到的上下文回答问题"
        )
        
        # English RAG Query with History Template
        rag_history_en = PromptConfig(
            name="rag_query_with_history",
            prompt_type=PromptType.RAG_QUERY_WITH_HISTORY,
            language=Language.ENGLISH,
            template="""You are a helpful AI assistant. Use the following context and conversation history to answer the user's question.

Conversation History:
{history}

Context:
{context}

Current Question: {query}

Instructions:
- Consider the conversation history for context and continuity
- Use the provided context to answer the current question
- Maintain consistency with previous responses
- If you cannot answer based on the available information, state this clearly
- Reference previous parts of the conversation when relevant

Answer:""",
            input_variables=["query", "context", "history"],
            description="RAG query template with conversation history support"
        )
        
        # Chinese RAG Query with History Template
        rag_history_zh = PromptConfig(
            name="rag_query_with_history",
            prompt_type=PromptType.RAG_QUERY_WITH_HISTORY,
            language=Language.CHINESE,
            template="""你是一个有用的AI助手。请根据以下上下文和对话历史回答用户的问题。

对话历史：
{history}

上下文：
{context}

当前问题：{query}

指示：
- 考虑对话历史以获得上下文和连续性
- 使用提供的上下文回答当前问题
- 与之前的回答保持一致性
- 如果你无法基于可用信息回答，请明确说明
- 在相关时引用对话的之前部分

答案：""",
            input_variables=["query", "context", "history"],
            description="支持对话历史的RAG查询模板"
        )
        
        # Summarization Template
        summarization_en = PromptConfig(
            name="summarization",
            prompt_type=PromptType.SUMMARIZATION,
            language=Language.ENGLISH,
            template="""Please provide a concise summary of the following text. Focus on the main points and key information.

Text to summarize:
{text}

Instructions:
- Capture the essential information and main ideas
- Keep the summary concise but informative
- Maintain the original meaning and context
- Use clear and simple language

Summary:""",
            input_variables=["text"],
            description="Template for text summarization"
        )
        
        # Query Rewriting Template
        query_rewriting_en = PromptConfig(
            name="query_rewriting",
            prompt_type=PromptType.QUERY_REWRITING,
            language=Language.ENGLISH,
            template="""Rewrite the following query to make it more specific and suitable for information retrieval. Consider the conversation context if provided.

Original Query: {query}

Context (if available): {context}

Instructions:
- Make the query more specific and detailed
- Add relevant keywords that might help in retrieval
- Maintain the original intent
- Consider synonyms and related terms
- Keep it as a question or search query

Rewritten Query:""",
            input_variables=["query", "context"],
            description="Template for query rewriting and expansion"
        )
        
        # Fact Checking Template
        fact_checking_en = PromptConfig(
            name="fact_checking",
            prompt_type=PromptType.FACT_CHECKING,
            language=Language.ENGLISH,
            template="""Please fact-check the following statement against the provided context. Determine if the statement is supported, contradicted, or cannot be verified by the context.

Statement to check: {statement}

Context:
{context}

Instructions:
- Compare the statement with the information in the context
- Determine if the statement is: SUPPORTED, CONTRADICTED, or UNVERIFIABLE
- Provide specific evidence from the context
- Explain your reasoning clearly

Fact Check Result:""",
            input_variables=["statement", "context"],
            description="Template for fact-checking statements against context"
        )
        
        # Register templates
        templates_to_register = [
            (rag_query_en, RAGQueryTemplate),
            (rag_query_zh, RAGQueryTemplate),
            (rag_history_en, RAGQueryWithHistoryTemplate),
            (rag_history_zh, RAGQueryWithHistoryTemplate),
            (summarization_en, SummarizationTemplate),
            (query_rewriting_en, BasePromptTemplate),
            (fact_checking_en, BasePromptTemplate),
        ]
        
        for config, template_class in templates_to_register:
            self.register_template(config, template_class)
    
    def register_template(self, config: PromptConfig, template_class: type = BasePromptTemplate):
        """Register a new prompt template."""
        if config.prompt_type not in self.templates:
            self.templates[config.prompt_type] = {}
        
        template_instance = template_class(config)
        self.templates[config.prompt_type][config.language] = template_instance
        
        logger.info(f"Registered template: {config.name} ({config.language})")
    
    def get_template(
        self, 
        prompt_type: PromptType, 
        language: Language = None
    ) -> BasePromptTemplate:
        """Get a prompt template by type and language."""
        language = language or self.default_language
        
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        if language not in self.templates[prompt_type]:
            # Fallback to English if requested language not available
            if Language.ENGLISH in self.templates[prompt_type]:
                logger.warning(f"Language {language} not available for {prompt_type}, using English")
                language = Language.ENGLISH
            else:
                raise ValueError(f"No template available for {prompt_type} in {language}")
        
        return self.templates[prompt_type][language]
    
    def format_prompt(
        self, 
        prompt_type: PromptType, 
        language: Language = None,
        **kwargs
    ) -> str:
        """Format a prompt with the given variables."""
        template = self.get_template(prompt_type, language)
        return template.format(**kwargs)
    
    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of available templates by type and language."""
        available = {}
        for prompt_type, lang_templates in self.templates.items():
            available[prompt_type] = list(lang_templates.keys())
        return available
    
    def load_templates_from_file(self, file_path: Union[str, Path]):
        """Load templates from a JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Template file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            for template_data in templates_data:
                config = PromptConfig(**template_data)
                
                # Determine template class based on type
                template_class_map = {
                    PromptType.RAG_QUERY: RAGQueryTemplate,
                    PromptType.RAG_QUERY_WITH_HISTORY: RAGQueryWithHistoryTemplate,
                    PromptType.SUMMARIZATION: SummarizationTemplate,
                }
                
                template_class = template_class_map.get(config.prompt_type, BasePromptTemplate)
                self.register_template(config, template_class)
            
            logger.info(f"Loaded {len(templates_data)} templates from {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error loading templates from {file_path}: {e}")
    
    def save_templates_to_file(self, file_path: Union[str, Path]):
        """Save current templates to a JSON file."""
        file_path = Path(file_path)
        
        templates_data = []
        for prompt_type, lang_templates in self.templates.items():
            for language, template in lang_templates.items():
                config_dict = {
                    'name': template.config.name,
                    'prompt_type': template.config.prompt_type,
                    'language': template.config.language,
                    'template': template.config.template,
                    'input_variables': template.config.input_variables,
                    'description': template.config.description,
                    'examples': template.config.examples,
                    'metadata': template.config.metadata,
                }
                templates_data.append(config_dict)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(templates_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(templates_data)} templates to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving templates to {file_path}: {e}")
    
    def create_chat_prompt(
        self,
        prompt_type: PromptType,
        language: Language = None,
        system_message: Optional[str] = None
    ) -> ChatPromptTemplate:
        """Create a chat prompt template for conversational models."""
        template = self.get_template(prompt_type, language)
        
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(SystemMessagePromptTemplate.from_template(system_message))
        
        # Add the main template as human message
        messages.append(HumanMessagePromptTemplate.from_template(template.config.template))
        
        return ChatPromptTemplate.from_messages(messages)
    
    def get_template_info(self, prompt_type: PromptType, language: Language = None) -> Dict[str, Any]:
        """Get information about a template."""
        template = self.get_template(prompt_type, language)
        
        return {
            'name': template.config.name,
            'prompt_type': template.config.prompt_type,
            'language': template.config.language,
            'input_variables': template.config.input_variables,
            'description': template.config.description,
            'examples': template.config.examples,
            'metadata': template.config.metadata,
        }
    
    def validate_template(self, prompt_type: PromptType, language: Language = None, **kwargs) -> bool:
        """Validate that a template can be formatted with given variables."""
        try:
            template = self.get_template(prompt_type, language)
            template.validate_inputs(**kwargs)
            return True
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False
    
    def set_default_language(self, language: Language):
        """Set the default language for templates."""
        self.default_language = language
        logger.info(f"Set default language to: {language}")


# Global template manager instance
template_manager = PromptTemplateManager()


class PromptBuilder:
    """Builder class for constructing complex prompts."""
    
    def __init__(self, template_manager: PromptTemplateManager = None):
        """
          Init   function implementation.
        """
        self.template_manager = template_manager or template_manager
        self.components = []
        self.variables = {}
    
    def add_system_message(self, message: str) -> 'PromptBuilder':
        """Add a system message."""
        self.components.append(('system', message))
        return self
    
    def add_template(self, prompt_type: PromptType, language: Language = None, **kwargs) -> 'PromptBuilder':
        """Add a template component."""
        self.components.append(('template', prompt_type, language, kwargs))
        return self
    
    def add_context(self, context: str) -> 'PromptBuilder':
        """Add context information."""
        self.variables['context'] = context
        return self
    
    def add_history(self, history: str) -> 'PromptBuilder':
        """Add conversation history."""
        self.variables['history'] = history
        return self
    
    def add_variable(self, key: str, value: Any) -> 'PromptBuilder':
        """Add a variable."""
        self.variables[key] = value
        return self
    
    def build(self) -> str:
        """Build the final prompt."""
        prompt_parts = []
        
        for component in self.components:
            if component[0] == 'system':
                prompt_parts.append(f"System: {component[1]}")
            elif component[0] == 'template':
                prompt_type, language, kwargs = component[1], component[2], component[3]
                # Merge builder variables with component variables
                merged_vars = {**self.variables, **kwargs}
                formatted_prompt = self.template_manager.format_prompt(
                    prompt_type, language, **merged_vars
                )
                prompt_parts.append(formatted_prompt)
        
        return '\n\n'.join(prompt_parts)
    
    def reset(self) -> 'PromptBuilder':
        """Reset the builder."""
        self.components.clear()
        self.variables.clear()
        return self


def create_rag_prompt(
    query: str,
    context: str,
    history: Optional[str] = None,
    language: Language = Language.ENGLISH
) -> str:
    """Convenience function to create RAG prompts."""
    if history:
        return template_manager.format_prompt(
            PromptType.RAG_QUERY_WITH_HISTORY,
            language,
            query=query,
            context=context,
            history=history
        )
    else:
        return template_manager.format_prompt(
            PromptType.RAG_QUERY,
            language,
            query=query,
            context=context
        )


def create_summarization_prompt(text: str, language: Language = Language.ENGLISH) -> str:
    """Convenience function to create summarization prompts."""
    return template_manager.format_prompt(
        PromptType.SUMMARIZATION,
        language,
        text=text
    )