"""
User Interface Module for Technical Document RAG Demo
=========================================

This module contains UI components and layouts for the Streamlit application.
It handles the rendering of different interface elements such as:
- Chat interface and message display
- Sidebar controls and status indicators
- Performance metrics visualisation
- File upload and processing controls

The UI components are designed to work with the state management handled in main.py.

Usage:
    This module is typically imported by main.py to render different parts of the UI.
    Components are organised as functions that can be called at appropriate points
    in the application flow.
"""

import os
import streamlit as st
import traceback
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import configuration
from config import (
    MAIN_LOGO_PATH, SIDEBAR_ICON_PATH, MAIN_PATH,
    UPLOAD_DIR, CACHE_DIR, FAISS_INDEX_DIR
)

def setup_page_config():
    """
    Configures the Streamlit page settings.
    """
    st.set_page_config(page_title="Technical Document RAG Demo", layout="wide")

def render_header():
    """
    Renders the application header with logo and title.
    """
    header_container = st.container()
    with header_container:
        title_col1, title_col2 = st.columns([1, 5])
        with title_col1:
            if MAIN_LOGO_PATH.exists():
                st.image(str(MAIN_LOGO_PATH), width=120)
            else:
                # Default image if logo not found
                st.image("https://img.freepik.com/vecteurs-libre/ingenieur-plat-organique-travaillant-construction_52683-59203.jpg", width=100)
        with title_col2:
            st.title("A Technical-Document-RAG Demo")

def render_metrics(performance_metrics: Dict[str, Any]):
    """
    Displays performance metrics in a multi-column layout.
    
    Args:
        performance_metrics (Dict[str, Any]): Dictionary containing performance measurements.
    """
    metrics_container = st.container()
    with metrics_container:
        st.caption("Pipeline Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Elements", performance_metrics.get('elements_count', 0))
            st.metric("Durée du Parsing:", f"{performance_metrics.get('parsing_time', 0):.2f}s")
        with metrics_col2:
            st.metric("Chunks", performance_metrics.get('nodes_count', 0))
            st.metric("Durée du process d'Embedding/Indexation)", f"{performance_metrics.get('indexing_time', 0):.2f}s")
        with metrics_col3:
            st.metric("Durée Totale du Pipeline", f"{performance_metrics.get('total_processing_time', 0):.2f}s")

def setup_sidebar():
    """
    Configures the sidebar with controls and displays the sidebar icon.
    
    Returns:
        tuple: (uploaded_file, similarity_top_k, rerank_top_n, force_reprocess)
    """
    st.sidebar.header("Panneau de Contrôle")
    if SIDEBAR_ICON_PATH.exists():
        st.sidebar.image(str(SIDEBAR_ICON_PATH), width=200)
    else:
        # Default image if icon not found
        st.sidebar.image(MAIN_LOGO_PATH if MAIN_LOGO_PATH.exists() else "https://img.freepik.com/vecteurs-libre/ingenieur-plat-organique-travaillant-construction_52683-59203.jpg", width=100)

    # File upload with validation
    uploaded_file = st.sidebar.file_uploader("Votre PDF ICI", type="pdf", key="pdf_uploader")
    
    # Control parameters
    similarity_top_k = st.sidebar.slider(
        "Nodes Retrieved (K)", 
        min_value=3, 
        max_value=20,
        value=10, 
        step=1, 
        key="top_k",
        help="Number of documents chunks initially retrieved based on similarity before re-ranking."
    )
    
    rerank_top_n = st.sidebar.slider(
        "Nodes Reranked (N)", 
        min_value=1, 
        max_value=10,
        value=3, 
        step=1, 
        key="top_n", 
        help="Number of documents chunks kept after re-ranking for relevance"
    )
    
    # Control buttons
    force_reprocess = st.sidebar.button(
        "🔄 Nettoyer les caches",
        key="force_reindex_button", 
        help="Forces re-parsing and re-indexing for the currently loaded document",
        type="secondary"
    )
    
    # Add separator
    st.sidebar.markdown("---")
    
    # Clear chat button
    clear_chat = st.sidebar.button(
        "🔄 Nettoyer la discussion", 
        type="secondary", 
        key="clear_chat_button",
        help="Clears the current chat history."
    )
    
    # Reset system button
    reset_system = st.sidebar.button(
        "❌ Réinitialiser le système", 
        type="primary", 
        key="reset_system_button",
        help="Clears current document, cache, index, and chat."
    )
    
    return uploaded_file, similarity_top_k, rerank_top_n, force_reprocess, clear_chat, reset_system

def display_welcome_screen():
    """
    Displays the welcome screen when no document is loaded.
    """
    col1, col2 = st.columns([1,2])
    with col1: 
        if MAIN_PATH.exists():
            st.image(MAIN_PATH, width=300)
        else:
            st.image("https://img.freepik.com/free-vector/file-searching-concept-illustration_114360-4690.jpg", width=300)
    with col2:
        st.subheader("Welcome to the Technical Document RAG Demo!")
        st.write("""
        1. Pour commencer, sur le `Panneau de Contrôle` uploadez un document technique de votre choix au format PDF.
                 
        2. Puis, cliquez sur `Demarrez`. Notre système se chargera de le parser, puis d'en générer des embeddings tout en l'indexant:
                 
        3. Adjustez les paramètres de retrieval dans le `Panneau de Contrôle` selon votre convenance(Optionel).
                 
        4. Posez les questions de votre choix, en précisant éventuellement le format de réponse souhaité.
        
        5. Notre système analysera le document, et vous fournira des réponses pertinentes, et surtout.....sourcées.
        - VOILA ! 
        - `Vous avez maintenant un assistant intelligent pour vos documents techniques`.
        """)
        st.caption("Rendez-vous sur le Panneau de contrôle à gauche pour commencer.")
         
    # Add welcome message
    for _ in range(5):
        st.write("")
        
    # Footer with powered by message
    st.divider()
    st.markdown(f"""
    <div style='text-align: center; margin-top: 20px; color: grey;'>
        Powered by LlamaIndex, Voyage AI, Claude AI, FAISS & Streamlit.
    </div>
    """, unsafe_allow_html=True)

def display_document_info(filename: str):
    """
    Displays information about the current document.
    
    Args:
        filename (str): The name of the uploaded file.
    """
    st.markdown(f"**Document Source**: '{filename}'")

def render_processing_status(parsing_done: bool, indexing_done: bool, query_engine: Any, processing_error: Optional[str] = None):
    """
    Renders the current processing status indicators.
    
    Args:
        parsing_done (bool): Whether parsing is complete.
        indexing_done (bool): Whether indexing is complete.
        query_engine (Any): The query engine object or None.
        processing_error (Optional[str]): Error message, if any.
    
    Returns:
        tuple: (parsing_status_placeholder, indexing_status_placeholder, query_engine_status_placeholder)
    """
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1: 
        parsing_status_placeholder = st.empty()
        if parsing_done: 
            parsing_status_placeholder.success("✅ Parsé avec succès")
        else: 
            parsing_status_placeholder.warning("⏳ Parsing en cours")
    
    with col2: 
        indexing_status_placeholder = st.empty()
        if indexing_done: 
            indexing_status_placeholder.success("✅ Indexé avec succès")
        else: 
            indexing_status_placeholder.warning("⏳ Indexation en cours")
    
    with col3: 
        query_engine_status_placeholder = st.empty()
        if query_engine: 
            query_engine_status_placeholder.success("✅ Moteur de requête prêt")
        elif processing_error: 
            query_engine_status_placeholder.error(f"❌ Failed: {processing_error}")
        else: 
            query_engine_status_placeholder.warning("⏳ Mise en place du moteur de requête")
    
    return parsing_status_placeholder, indexing_status_placeholder, query_engine_status_placeholder

def render_chat_interface(chat_history: List[Dict[str, str]]):
    """
    Renders the chat interface with user and assistant messages.
    
    Args:
        chat_history (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
        
    Returns:
        tuple: (chat_container, user_query, send_button)
    """
    st.divider()
    st.subheader("Posez vos questions:")

    # Chat message display area
    chat_container = st.container(height=500, border=False)

    # Use the 'with' statement to add elements inside the container
    with chat_container:
        for message in chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                # Display user messages using custom HTML/CSS
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='
                        background-color: rgba(0, 255, 0, 0.1); 
                        color: #FFFFFF;
                        padding: 10px 15px; 
                        border-radius: 15px 15px 0 15px;
                        max-width: 70%;
                        border: 1px solid rgba(0, 255, 0, 0.15); 
                        box-shadow: 0 1px 1px rgba(0,0,0,0.1); 
                        '>
                        <p style='margin: 0;'><strong>🧑‍💼 You:</strong><br>{content}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif role == "assistant":
                # Display assistant messages using custom HTML/CSS
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='
                        background-color: rgba(0, 0, 255, 0.1); 
                        color: #FFFFFF;
                        padding: 10px 15px;
                        border-radius: 15px 15px 15px 0; 
                        max-width: 70%; 
                        border: 1px solid #E0E0E0;
                        box-shadow: 0 1px 1px rgba(0,0,0,0.1); 
                        '>
                        <p style='margin: 0;'><strong>🤖 Assistant:</strong><br>{content}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif role == "system":
                # Handle 'system' messages based on their content
                if "Sources (" in content:
                    # Sources information in collapsible expander
                    with st.expander("📚 View Sources", expanded=False):
                        st.markdown(content, unsafe_allow_html=True)
                elif "Error:" in content:
                    # Error messages
                    st.error(content.replace("Error: ",""), icon="❌")
                else:
                    # Other system messages
                    st.info(content)
   
    # Input area at the bottom
    st.divider()
    input_container = st.container()
    with input_container:
        # Define an initial empty value if the button has been previously pressed 
        initial_value = "" if st.session_state.get("submit_pressed", False) else st.session_state.get("query_input", "")
        
        # Text area for user input        
        user_query = st.text_area("Votre question ici :", value=initial_value, key="query_input", height=100)#
        send_button = st.button(
            "GO", 
            key="get_answer_button", 
            icon="✅", 
            type="secondary", 
            help="Envoyer la question au moteur de requête."
        )
    
    return chat_container, user_query, send_button

def get_cache_path_for_file(filename: str) -> Path:
    """
    Helper function to generate cache path for a specific file.
    
    Args:
        filename (str): The name of the uploaded file.
        
    Returns:
        Path: The path to the cache file.
    """
    file_stem = Path(filename).stem
    sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)
    return CACHE_DIR / f"{sanitized_stem}_parsed_cache.pkl"

def get_index_dir_for_file(filename: str) -> Path:
    """
    Helper function to generate index directory path for a specific file.
    
    Args:
        filename (str): The name of the uploaded file.
        
    Returns:
        Path: The path to the index directory.
    """
    file_stem = Path(filename).stem
    sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)
    return FAISS_INDEX_DIR / sanitized_stem
