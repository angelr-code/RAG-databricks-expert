import streamlit as st
import requests
import json
import os
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Databricks Expert RAG",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
API_BASE_URL = st.secrets("API_URL")
STREAM_ENDPOINT = f"{API_BASE_URL}/query/stream"
GENERATE_ENDPOINT = f"{API_BASE_URL}/query/generate"

# Databricks Brand Colors (DARK MODE VERSION)
COLORS = {
    "primary": "#FF3621",       # Databricks Red
    "secondary": "#1B3139",     # Deep Navy Background
    "accent": "#2D3E46",        # Lighter Navy for cards/sidebar
    "text": "#F5F5F5",          # Near White
    "text_secondary": "#B0B8BF",# Light Grey
    "sidebar": "#15252B",       # Darker Sidebar
    "border": "#3A4B53",        # Subtle Border
    "success": "#00CC88",       # Bright Success
    "warning": "#FFB03B"        # Bright Warning
}

# Model Configurations
@dataclass
class ModelInfo:
    id: str
    name: str
    icon: str
    description: str

OPENROUTER_MODELS = [
    ModelInfo("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B Instruct", "🦙", "High-quality reasoning and fast"),
    ModelInfo("tngtech/deepseek-r1t2-chimera:free", "Deepseek-r1t2-chimera", "🐳", "Massive 671B reasoning & coding engine. Maximum performance"),
    ModelInfo("nvidia/nemotron-nano-12b-v2-vl:free", "Nvidia Nemotron 12B", "🟢", "Light, Ultra fast and efficient")
]

OPENAI_MODELS = [
    ModelInfo("gpt-5-nano", "GPT-5 Nano", "🚀", "Maximum performance at minimal cost"),
    ModelInfo("gpt-4o-mini", "GPT-4o Mini", "⚡", "Fast, affordable, and highly efficient")
]

OPENAI_PRICING = {
    "gpt-5-nano": {"input": 0.050, "output": 0.400},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600}
}

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def clean_and_format_response(text: str) -> str:
    """
    Cleans and formats the raw text response from the LLM for better UI rendering.
    
    This function performs lightweight post-processing to ensure Markdown compatibility
    without aggressively altering the stream structure, preventing the concatenation
    issues caused by regex overuse.

    Args:
        text (str): The raw text string to be processed.

    Returns:
        str: The cleaned and formatted text string ready for display.
    """
    if not text:
        return text
    
    # 1. Remove duplicate citation references like [1][2][3] if present
    text = re.sub(r'(\[\d+\])+', '', text)
    
    # 2. Ensure headers (###) have preceding spacing for valid Markdown rendering
    # Only applies if a newline isn't already present before the header
    text = re.sub(r'([^\n])\n(#{2,3}\s)', r'\1\n\n\2', text)
    
    # 3. Trim leading/trailing whitespace
    return text.strip()

# ============================================================================
# CSS STYLES
# ============================================================================

def inject_custom_css():
    """Inject custom Databricks styles (Dark Mode)"""
    st.markdown(f"""
    <style>
        /* Global App Styles */
        .stApp {{
            background-color: {COLORS['secondary']};
            color: {COLORS['text']};
        }}
        
        /* Header */
        .main-header {{
            text-align: center;
            padding: 2rem 0 1rem 0;
            background: linear-gradient(180deg, #24333A 0%, {COLORS['secondary']} 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .main-header h1 {{
            color: white !important;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 10px rgba(255, 54, 33, 0.3);
        }}
        
        .main-header p {{
            color: {COLORS['text_secondary']} !important;
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['sidebar']};
            border-right: 1px solid {COLORS['border']};
        }}
        
        [data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}

        /* Sidebar Text */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: white !important;
        }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
            color: {COLORS['text_secondary']} !important;
        }}
        
        /* Chat Messages Container */
        .stChatMessage {{
            background-color: {COLORS['accent']};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        [data-testid="stChatMessage"] {{
            background-color: {COLORS['accent']};
        }}

        /* ============================================================ */
        /* TOTAL TYPOGRAPHY CORRECTION WITHIN CHAT */
        /* ============================================================ */
        
        /* Full reset of text styles in chat messages */
        [data-testid="stChatMessage"] * {{
            font-weight: normal !important;
        }}
        
        /* H1 Headers */
        [data-testid="stChatMessage"] h1 {{
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin: 1.2rem 0 0.8rem 0 !important;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid {COLORS['border']};
            color: {COLORS['text']} !important;
        }}
        
        /* H2 Headers */
        [data-testid="stChatMessage"] h2 {{
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin: 1rem 0 0.6rem 0 !important;
            color: {COLORS['primary']} !important;
        }}
        
        /* H3 Headers */
        [data-testid="stChatMessage"] h3 {{
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin: 0.8rem 0 0.5rem 0 !important;
            color: {COLORS['text']} !important;
        }}
        
        /* Paragraphs - NORMAL SIZE AND NO BOLD */
        [data-testid="stChatMessage"] p {{
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            margin-bottom: 0.8rem !important;
            font-weight: 400 !important;
            color: {COLORS['text']} !important;
        }}
        
        /* Text inside divs */
        [data-testid="stChatMessage"] div {{
            font-size: 0.95rem !important;
            font-weight: 400 !important;
            color: {COLORS['text']} !important;
        }}
        
        /* Lists */
        [data-testid="stChatMessage"] ul,
        [data-testid="stChatMessage"] ol {{
            margin: 0.8rem 0;
            padding-left: 1.5rem;
            font-size: 0.95rem !important;
            font-weight: 400 !important;
        }}
        
        [data-testid="stChatMessage"] li {{
            margin-bottom: 0.4rem;
            line-height: 1.6;
            font-weight: 400 !important;
            color: {COLORS['text']} !important;
        }}
        
        /* Strong/Bold - With moderate weight */
        [data-testid="stChatMessage"] strong {{
            color: {COLORS['primary']} !important;
            font-weight: 600 !important;
        }}
        
        /* Inline code */
        [data-testid="stChatMessage"] code {{
            background-color: {COLORS['sidebar']};
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            font-weight: 400 !important;
            color: #00CC88 !important;
        }}
        
        /* Code blocks */
        [data-testid="stChatMessage"] pre {{
            background-color: {COLORS['sidebar']};
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid {COLORS['border']};
            font-weight: 400 !important;
        }}
        
        [data-testid="stChatMessage"] pre code {{
            background-color: transparent;
            padding: 0;
        }}

        /* ============================================================ */
        
        /* Links */
        a {{
            color: {COLORS['primary']} !important;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s ease;
        }}
        
        a:hover {{
            color: #FF6B5B !important;
            text-decoration: underline;
        }}
        
        /* Buttons */
        .stButton > button {{
            border: 1px solid {COLORS['primary']};
            color: white !important;
            background-color: {COLORS['primary']};
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
            width: 100%;
        }}
        
        .stButton > button:hover {{
            background-color: #D92D1B;
            border-color: #D92D1B;
            box-shadow: 0 0 15px rgba(255, 54, 33, 0.4);
            transform: translateY(-1px);
        }}

        .stButton > button:focus {{
            color: white !important;
        }}
        
        /* Inputs */
        .stChatInput textarea, .stTextInput input {{
            background-color: {COLORS['sidebar']} !important;
            color: white !important;
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
        }}
        
        .stChatInput textarea:focus, .stTextInput input:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 1px {COLORS['primary']} !important;
        }}

        /* Selectbox */
        div[data-baseweb="select"] > div {{
            background-color: {COLORS['sidebar']} !important;
            color: white !important;
            border-color: {COLORS['border']} !important;
        }}
        
        /* Cards (Info/Warning/Pricing) */
        .info-card {{
            background-color: rgba(255, 54, 33, 0.1);
            border-left: 4px solid {COLORS['primary']};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
        }}
        
        .pricing-card {{
            background-color: rgba(0, 204, 136, 0.1);
            border-left: 4px solid {COLORS['success']};
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: {COLORS['text']};
        }}
        
        .warning-card {{
            background-color: rgba(255, 176, 59, 0.1);
            border-left: 4px solid {COLORS['warning']};
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: {COLORS['text']};
        }}
        
        /* Sources */
        .source-container {{
            background-color: {COLORS['sidebar']};
            padding: 0.75rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border: 1px solid {COLORS['border']};
            transition: all 0.2s ease;
        }}
        
        .source-container:hover {{
            background-color: {COLORS['accent']};
            border-color: {COLORS['primary']};
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {COLORS['sidebar']} !important;
            color: white !important;
            border-radius: 8px;
        }}
        
        /* Loading Text */
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .loading-text {{
            animation: pulse 1.5s ease-in-out infinite;
            color: {COLORS['primary']};
            font-weight: 600;
        }}

        /* Radio Buttons */
        .stRadio label {{
            color: {COLORS['text']} !important;
        }}
        
        div[data-testid="InputInstructions"] {{
            display: none !important;
        }}
        
        div[data-testid="stTextInput"] input {{
            padding-right: 40px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🧱 Databricks Expert Assistant</h1>
        <p>Technical Assistant with RAG on Databricks Official Documentation</p>
    </div>
    """, unsafe_allow_html=True)

def render_model_selector(models: List[ModelInfo], key_prefix: str) -> ModelInfo:
    """Render model selector with visual cards"""
    options = {f"{m.icon} {m.name}": m for m in models}
    
    selected_display = st.selectbox(
        "Select Model",
        options=options.keys(),
        key=f"{key_prefix}_model_select"
    )
    
    selected_model = options[selected_display]
    
    # Show description of selected model
    st.markdown(f"""
    <div class="info-card">
        <strong>{selected_model.icon} {selected_model.name}</strong><br>
        <small style="color: #D3D3D3;">{selected_model.description}</small>
    </div>
    """, unsafe_allow_html=True)
    
    return selected_model

def render_pricing_info(model_id: str):
    """Render pricing info for OpenAI"""
    if model_id in OPENAI_PRICING:
        pricing = OPENAI_PRICING[model_id]
        st.markdown(f"""
        <div class="pricing-card">
            <strong> Pricing (per 1M tokens)</strong><br>
            <span style="color: #D3D3D3;">• Input: ${pricing['input']}<br>
            • Output: ${pricing['output']}</span>
        </div>
        """, unsafe_allow_html=True)

def render_sources(sources: List[str]):
    """Render list of consulted sources"""
    if not sources:
        return
    
    with st.expander("Sources Consulted", expanded=False):
        for idx, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-container">
                <strong>{idx}.</strong> <a href="{source}" target="_blank" style="word-break: break-all;">{source}</a>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# BUSINESS LOGIC
# ============================================================================

def initialize_session_state():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": """Hello! 👋 I am your **specialized Databricks AWS technical assistant**.

I can help you with:
- 🔧 **Delta Table Optimization**
- 🔐 **Unity Catalog & Data Governance**
- ⚙️ **Databricks Jobs & Workflows**
- ⚡ **PySpark & Structured Streaming**
- 📊 **MLflow & Machine Learning**

How can I help you today?

**‼️ NOTE:** This app runs in AWS Lambda, first request may take longer due to cold start.
""",
            "sources": []
        }]

def stream_api_response(
    query: str,
    provider: str,
    model_id: str,
    limit: int,
    placeholder,
    api_key: Optional[str] = None
) -> Tuple[str, List[str], Optional[str]]:
    """
    Executes a streaming POST request to the backend API and updates the UI in real-time.

    This function handles the response stream byte-by-byte to preserve original formatting
    (newlines, indentation) which acts as a fix for 'iter_lines' formatting issues.
    It also parses the first line of the stream to extract source metadata.

    Args:
        query (str): The user's search query.
        provider (str): The AI provider to use (e.g., "openai", "openrouter").
        model_id (str): The specific model identifier.
        limit (int): Number of source documents to retrieve.
        placeholder (streamlit.empty): The Streamlit placeholder to update with streaming text.
        api_key (Optional[str]): The API key for the provider, if required.

    Returns:
        Tuple[str, List[str], Optional[str]]: A tuple containing:
            - The full generated response text.
            - A list of source URLs/citations.
            - An error message string if the request failed, else None.
    """
    headers = {"Content-Type": "application/json"}
    backend_secret = st.secrets("BACKEND_SECRET")
    if backend_secret:
        headers["X-Backend-Secret"] = backend_secret
    if api_key:
        headers["OpenAI-API-Key"] = api_key
    
    payload = {
        "query_text": query,
        "provider": provider,
        "model": model_id,
        "limit": limit
    }
    
    try:
        with requests.post(STREAM_ENDPOINT, json=payload, headers=headers, stream=True, timeout=60) as response:
            if response.status_code != 200:
                error_msg = f"Server error ({response.status_code})"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail.get('detail', response.text)}"
                except:
                    error_msg += f": {response.text[:200]}"
                return "", [], error_msg
            
            full_response = ""
            sources = []
            buffer = ""
            is_first_chunk = True
            
            # Use iter_content to read bytes directly, preserving all whitespace and newlines
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode('utf-8')
                    
                    # Logic to handle the first line which contains JSON metadata (sources)
                    if is_first_chunk:
                        buffer += text_chunk
                        if '\n' in buffer:
                            # Split only on the first newline to separate metadata from content
                            first_line, remaining = buffer.split('\n', 1)
                            try:
                                data = json.loads(first_line)
                                if data.get("type") == "sources":
                                    sources = data.get("data", [])
                            except json.JSONDecodeError:
                                # If decode fails, treat first line as part of the content
                                full_response += first_line + "\n"
                            
                            # Append the rest of the buffer to the response and display
                            full_response += remaining
                            placeholder.markdown(full_response + "▌")
                            is_first_chunk = False
                        continue
                    
                    # Standard streaming processing
                    full_response += text_chunk
                    
                    # Update UI with the accumulated text
                    # We avoid heavy regex cleaning here to prevent visual jitter
                    placeholder.markdown(full_response + "▌")
            
            # Final render without the cursor
            final_text = clean_and_format_response(full_response)
            placeholder.markdown(final_text)
            
            return final_text, sources, None
            
    except requests.exceptions.ConnectionError:
        return "", [], "❌ Could not connect to backend. Check if it is running at localhost:8000"
    except requests.exceptions.Timeout:
        return "", [], "⏱️ Request timed out. Try a simpler query."
    except Exception as e:
        return "", [], f"❌ Unexpected error: {str(e)}"

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> Dict:
    """Render sidebar with configuration"""
    with st.sidebar:
        st.image("https://1000marcas.net/wp-content/uploads/2025/01/Databricks-Emblem.png", width=180)
        st.markdown("### Configuration")
        
        # Provider Selector
        provider_option = st.radio(
            "AI Provider",
            ["OpenRouter", "OpenAI"],
            captions=[
                "Free open source models",
                "Proprietary models (requires API key)"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Configuration Variables
        config = {
            "api_key": None,
            "model_id": "",
            "provider": "",
            "limit": 10,
            "use_context": False
        }
        
        # Configuration based on provider
        if "OpenRouter" in provider_option:
            config["provider"] = "OpenRouter"
            
            st.markdown("""
            <div class="pricing-card" style="background-color: rgba(0, 204, 136, 0.1);">
                <strong style="color: #00CC88;">OpenRouter Selected</strong><br>
                <small>API key managed internally</small>
            </div>
            """, unsafe_allow_html=True)
            
            selected_model = render_model_selector(OPENROUTER_MODELS, "openrouter")
            config["model_id"] = selected_model.id
            
        else:  # OpenAI
            config["provider"] = "openai"
            
            st.markdown("""
            <div class="warning-card">
                <strong>🔑 Bring Your Own Key</strong><br>
                <small>Your key is sent securely via HTTPS</small>
            </div>
            """, unsafe_allow_html=True)
            
            # API Key Form
            with st.form(key="openai_key_form"):
                api_key_input = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-proj-...",
                    help="Your key is not stored on the server"
                )
                
                submitted = st.form_submit_button("Authenticate", use_container_width=True)
            
            config["api_key"] = api_key_input
            
            if api_key_input:
                st.caption("🔐 Key loaded in session")
            
            selected_model = render_model_selector(OPENAI_MODELS, "openai")
            config["model_id"] = selected_model.id
            
            render_pricing_info(selected_model.id)
        
        st.markdown("---")
        
        # Additional Configuration
        st.subheader("Parameters")
        
        # Updated slider: Max 20, default 10
        config["limit"] = st.slider(
            "Sources to retrieve",
            min_value=1,
            max_value=20, 
            value=10
        )
        
        # Context Window toggle
        config["use_context"] = st.toggle(
            "Enable Context Window",
            value=True,
            help="If enabled, this will smartly use previous messages to improve search accuracy for follow-up questions."
        )
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        return config

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application function"""
    # Setup
    inject_custom_css()
    initialize_session_state()
    render_header()
    
    # Configuration from sidebar
    config = render_sidebar()
    
    # OpenAI API Key Validation
    if config["provider"] == "openai" and not config["api_key"]:
        st.warning("⚠️ **Action Required:** Enter your OpenAI API Key in the sidebar to continue.")
        st.info("💡 If you don't have one, you can get it at [platform.openai.com](https://platform.openai.com)")
        st.stop()
    
    # Render chat history
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
        
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))
    
    # User Input
    if prompt := st.chat_input("Ask a question about Databricks..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            
            # --- START OF FULL CONTEXT LOGIC (USER + ASSISTANT) ---
            query_to_send = prompt
            
            # Check if we have enough history: 
            # We need at least 3 messages in state: [User_1, AI_1, User_2(Current)]
            if config["use_context"] and len(st.session_state.messages) >= 3:
                
                # Retrieve the actual last interaction
                # messages[-1] is the current prompt (we just appended it)
                # messages[-2] is the last Assistant response
                # messages[-3] is the last User prompt
                
                last_assistant_msg = st.session_state.messages[-2]["content"]
                last_user_msg = st.session_state.messages[-3]["content"]
                
                # Heuristics: Should we attach context?
                # Condition A: Short prompt
                is_short = len(prompt.split()) < 8
                
                # Condition B: Reference words
                ref_words = ['it', 'that', 'this', 'code', 'example', 'previous', 'last', 'one', 'explain', 'mean', 'why']
                has_reference = any(word in prompt.lower() for word in ref_words)
                
                if is_short or has_reference:
                    # We combine everything so the LLM sees exactly what happened
                    query_to_send = f"""
                    PREVIOUS INTERACTION:
                    User: {last_user_msg}
                    Assistant: {last_assistant_msg}
                    
                    CURRENT QUESTION:
                    {prompt}
                    """
            # --- END OF LOGIC ---

            with st.spinner("Searching documentation..."):
                full_response, sources, error = stream_api_response(
                    query=query_to_send, 
                    provider=config["provider"],
                    model_id=config["model_id"],
                    limit=config["limit"],
                    placeholder=response_placeholder,
                    api_key=config["api_key"]
                )
            
            if error:
                st.error(error)
                st.stop()
            
            render_sources(sources)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
            st.rerun()

if __name__ == "__main__":
    main()