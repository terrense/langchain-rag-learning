"""Streamlit main application."""

import streamlit as st
import requests
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Page config
st.set_page_config(
    page_title="LangChain RAG Learning",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_providers():
    """Get available providers from API."""
    try:
        response = requests.get("http://localhost:8000/providers", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def query_rag(question, provider="deepseek"):
    """Query the RAG system."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question, "provider": provider},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 LangChain RAG Learning Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Status
        api_status = check_api_status()
        if api_status:
            st.markdown('<p class="status-success">✅ API Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ API Disconnected</p>', unsafe_allow_html=True)
            st.error("Please start the API server first:\n`uvicorn src.langchain_rag_learning.api.main:app --reload --port 8000`")
        
        # Provider selection
        providers = get_providers()
        if providers:
            enabled_providers = [name for name, config in providers.items() if config.get("enabled", False)]
            if enabled_providers:
                selected_provider = st.selectbox(
                    "Select LLM Provider",
                    enabled_providers,
                    index=0 if "deepseek" not in enabled_providers else enabled_providers.index("deepseek")
                )
            else:
                st.warning("No providers enabled")
                selected_provider = "deepseek"
        else:
            selected_provider = "deepseek"
        
        st.markdown("---")
        
        # Quick links
        st.header("📚 Quick Links")
        st.markdown("""
        - [API Documentation](http://localhost:8000/docs)
        - [GitHub Repository](#)
        - [Quick Start Guide](QUICKSTART.md)
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "💬 Chat", "📖 Learning", "⚙️ Settings"])
    
    with tab1:
        st.header("Welcome to LangChain RAG Learning!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h3>🎯 What You'll Learn</h3>
                <ul>
                    <li>LangChain framework fundamentals</li>
                    <li>RAG (Retrieval-Augmented Generation) systems</li>
                    <li>Vector databases and embeddings</li>
                    <li>Production-ready LLM applications</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h3>🚀 Features</h3>
                <ul>
                    <li>Multi-LLM support (OpenAI, DeepSeek, local models)</li>
                    <li>Interactive learning modules</li>
                    <li>Document upload and processing</li>
                    <li>Real-time chat with your documents</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # System status
        st.header("📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_status:
                st.success("API Server: Running")
            else:
                st.error("API Server: Stopped")
        
        with col2:
            if providers:
                enabled_count = len([p for p in providers.values() if p.get("enabled", False)])
                st.info(f"Providers: {enabled_count} enabled")
            else:
                st.warning("Providers: Not loaded")
        
        with col3:
            st.info("Vector DB: Ready")
    
    with tab2:
        st.header("💬 Chat with RAG")
        
        if not api_status:
            st.error("Please start the API server to use the chat feature.")
            return
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_rag(prompt, selected_provider)
                    
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                        ai_response = "Sorry, I encountered an error processing your question."
                    else:
                        ai_response = response.get("answer", "No response received")
                        
                        # Show sources if available
                        if response.get("sources"):
                            st.markdown("**Sources:**")
                            for source in response["sources"]:
                                st.markdown(f"- {source}")
                    
                    st.markdown(ai_response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    with tab3:
        st.header("📖 Learning Modules")
        
        st.info("Learning modules are coming soon! This will include:")
        
        modules = [
            "🔤 LangChain Basics",
            "🧠 Understanding RAG",
            "📊 Vector Databases",
            "🔍 Retrieval Strategies",
            "⚡ Performance Optimization",
            "🚀 Production Deployment"
        ]
        
        for i, module in enumerate(modules, 1):
            with st.expander(f"Module {i}: {module}"):
                st.markdown(f"""
                **Status:** Coming Soon
                
                **What you'll learn:**
                - Theoretical foundations
                - Practical implementation
                - Best practices
                - Real-world examples
                
                **Duration:** ~30 minutes
                """)
    
    with tab4:
        st.header("⚙️ Settings")
        
        # Provider configuration
        st.subheader("LLM Providers")
        
        if providers:
            for name, config in providers.items():
                with st.expander(f"{config.get('name', name)} ({name})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {config.get('type', 'Unknown')}")
                        st.write(f"**Status:** {'✅ Enabled' if config.get('enabled') else '❌ Disabled'}")
                    
                    with col2:
                        models = config.get('models', [])
                        if models:
                            st.write("**Available Models:**")
                            for model in models:
                                st.write(f"- {model}")
        else:
            st.warning("No provider configuration loaded")
        
        # Environment info
        st.subheader("Environment")
        st.code(f"""
API URL: http://localhost:8000
Selected Provider: {selected_provider}
Streamlit Version: {st.__version__}
        """)

if __name__ == "__main__":
    main()