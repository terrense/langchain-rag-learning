#!/usr/bin/env python3
"""
Environment setup script for LangChain RAG Learning Project
This script helps you set up the conda environment and initial configuration.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def check_conda():
    """Check if conda is available."""
    result = run_command("conda --version", check=False)
    return result is not None and result.returncode == 0


def create_conda_environment():
    """Create conda environment from environment.yml."""
    print("Creating conda environment...")
    
    if not Path("environment.yml").exists():
        print("Error: environment.yml not found!")
        return False
    
    # Create environment
    result = run_command("conda env create -f environment.yml", check=False)
    if result and result.returncode == 0:
        print("✅ Conda environment created successfully!")
        return True
    else:
        print("❌ Failed to create conda environment")
        return False


def setup_environment_file():
    """Set up .env file from .env.example."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("Warning: .env.example not found")
        return False
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Copy .env.example to .env
    try:
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from .env.example")
        print("📝 Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/uploads",
        "data/chroma",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_development_tools():
    """Install development tools in the conda environment."""
    print("Installing development tools...")
    
    commands = [
        "conda run -n langchain-rag-learning pip install -e .",
        "conda run -n langchain-rag-learning pre-commit install"
    ]
    
    for command in commands:
        result = run_command(command, check=False)
        if result and result.returncode == 0:
            print(f"✅ Successfully ran: {command}")
        else:
            print(f"❌ Failed to run: {command}")


def main():
    """Main setup function."""
    print("🚀 Setting up LangChain RAG Learning Project Environment")
    print("=" * 60)
    
    # Check if conda is available
    if not check_conda():
        print("❌ Conda is not available. Please install Anaconda or Miniconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    
    print("✅ Conda is available")
    
    # Create conda environment
    if not create_conda_environment():
        print("❌ Failed to create conda environment")
        sys.exit(1)
    
    # Set up .env file
    setup_environment_file()
    
    # Create directories
    create_directories()
    
    # Install development tools
    install_development_tools()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup completed!")
    print("\n📋 Next steps:")
    print("1. Activate the environment: conda activate langchain-rag-learning")
    print("2. Edit .env file and add your API keys (especially DEEPSEEK_API_KEY)")
    print("3. Run the application: make run-api")
    print("4. In another terminal, run the UI: make run-ui")
    print("\n💡 Recommended: Get a DeepSeek API key (very affordable)")
    print("   Visit: https://platform.deepseek.com/")
    print("\n🔧 For local models (free), install Ollama:")
    print("   Visit: https://ollama.ai/")


if __name__ == "__main__":
    main()