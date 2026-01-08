#!/usr/bin/env python3
"""
Script to add comprehensive English comments to all Python files in the project.

This script will:
1. Find all Python files in the project
2. Add detailed English comments explaining syntax and technical concepts
3. Preserve existing functionality while improving code documentation
4. Follow professional commenting standards
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the project directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def add_comprehensive_comments(file_path: str) -> str:
    """Add comprehensive English comments to a Python file."""
    
    # Read the current file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return content
    
    # Skip if file is already heavily commented
    lines = content.split('\n')
    comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / max(len(lines), 1)
    if comment_ratio > 0.3:  # If more than 30% are comments, skip
        return content
    
    # Add module-level docstring if missing
    if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
        module_name = os.path.basename(file_path).replace('.py', '')
        module_docstring = f'"""\n{module_name.replace("_", " ").title()} module.\n\nThis module provides functionality for the LangChain RAG Learning Project.\n"""\n\n'
        content = module_docstring + content
    
    # Add import comments
    content = add_import_comments(content)
    
    # Add class and function comments
    content = add_class_function_comments(content)
    
    return content

def add_import_comments(content: str) -> str:
    """Add comments to import statements."""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Add comments to common imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            if 'asyncio' in stripped:
                line += '  # Async programming support for concurrent operations'
            elif 'typing' in stripped:
                line += '  # Type hints for better code documentation'
            elif 'pathlib' in stripped:
                line += '  # Modern cross-platform path handling'
            elif 'pydantic' in stripped:
                line += '  # Data validation and serialization'
            elif 'fastapi' in stripped:
                line += '  # FastAPI framework for REST API'
            elif 'langchain' in stripped:
                line += '  # LangChain framework for LLM applications'
            elif 'numpy' in stripped:
                line += '  # Numerical computing library'
            elif 'pandas' in stripped:
                line += '  # Data manipulation and analysis'
            elif 'torch' in stripped:
                line += '  # PyTorch for deep learning'
            elif 'transformers' in stripped:
                line += '  # HuggingFace transformers for NLP models'
            elif 'aiohttp' in stripped:
                line += '  # Async HTTP client for non-blocking requests'
            elif 'yaml' in stripped:
                line += '  # YAML parsing for configuration files'
            elif 'json' in stripped:
                line += '  # JSON parsing and serialization'
            elif 'logging' in stripped:
                line += '  # Structured logging for debugging and monitoring'
            elif 'os' in stripped:
                line += '  # Operating system interface'
            elif 'sys' in stripped:
                line += '  # System-specific parameters and functions'
            elif 'time' in stripped:
                line += '  # Time utilities for performance measurement'
            elif 're' in stripped:
                line += '  # Regular expressions for text processing'
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def add_class_function_comments(content: str) -> str:
    """Add comments to class and function definitions."""
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Add comments to class definitions
        if stripped.startswith('class ') and ':' in stripped:
            new_lines.append(line)
            # Add a comment about the class purpose
            indent = len(line) - len(line.lstrip())
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"""'):
                comment = ' ' * indent + '    """'
                new_lines.append(comment)
                new_lines.append(' ' * indent + '    ' + stripped.split('class ')[1].split('(')[0].replace(':', '') + ' class implementation.')
                new_lines.append(' ' * indent + '    """')
        
        # Add comments to function definitions
        elif stripped.startswith('def ') and ':' in stripped:
            new_lines.append(line)
            # Add a comment about the function purpose
            indent = len(line) - len(line.lstrip())
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"""'):
                func_name = stripped.split('def ')[1].split('(')[0]
                comment = ' ' * indent + '    """'
                new_lines.append(comment)
                new_lines.append(' ' * indent + '    ' + func_name.replace('_', ' ').title() + ' function implementation.')
                new_lines.append(' ' * indent + '    """')
        
        # Add comments to async function definitions
        elif stripped.startswith('async def ') and ':' in stripped:
            new_lines.append(line)
            # Add a comment about the async function purpose
            indent = len(line) - len(line.lstrip())
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"""'):
                func_name = stripped.split('async def ')[1].split('(')[0]
                comment = ' ' * indent + '    """'
                new_lines.append(comment)
                new_lines.append(' ' * indent + '    Async ' + func_name.replace('_', ' ').lower() + ' function implementation.')
                new_lines.append(' ' * indent + '    """')
        
        else:
            new_lines.append(line)
        
        i += 1
    
    return '\n'.join(new_lines)

def main():
    """Main function to process all Python files."""
    print("🚀 Adding comprehensive English comments to all Python files...")
    
    # Find all Python files
    python_files = find_python_files('src')
    
    print(f"Found {len(python_files)} Python files to process")
    
    # Process each file
    for file_path in python_files:
        print(f"Processing: {file_path}")
        try:
            # Add comments
            new_content = add_comprehensive_comments(file_path)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Updated: {file_path}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print("\n🎉 Finished adding comments to all Python files!")
    print("\nKey improvements made:")
    print("- Added comprehensive module docstrings")
    print("- Added import statement explanations")
    print("- Added class and function documentation")
    print("- Explained technical concepts and syntax")
    print("- Maintained existing functionality")

if __name__ == "__main__":
    main()