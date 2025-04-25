"""
Main Application Module for Technical Document RAG Demo
===========================================

This is the entry point for the Technical Document RAG Demo application.
It orchestrates all other modules and manages the application flow and state.

Key responsibilities:
- Application initialisation
- State management
- Processing workflow coordination
- Handling user interactions
- Error handling and recovery

"""

import os
import re
import time
import traceback
import logging
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import configuration
from config import (
    VOYAGE_API_KEY, ANTHROPIC_API_KEY, USE_GPU,
    UPLOAD_DIR, FAISS_AVAILABLE
)

# Import modules
from pdf_parser import PDFParser
from cache_utils import get_cache_path, save_parsed_elements, load_parsed_elements
from indexing import create_llama_nodes, setup_global_settings, build_faiss_index, load_faiss_index
from query_engine import setup_query_engine
import ui

# Set up module-level logger
logger = logging.getLogger(__name__)

def init_session_state():
    """
    Initialises or resets session state variables for the application.
    """
    # Basic state control
    st.session_state.setdefault('submit_pressed', False)
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Processing pipeline states
    st.session_state.setdefault('parsing_done', False)
    st.session_state.setdefault('indexing_done', False)
    st.session_state.setdefault('parsed_elements', None)
    st.session_state.setdefault('faiss_index', None)
    st.session_state.setdefault('query_engine', None)
    
    # File and paths tracking
    st.session_state.setdefault('uploaded_filename', None)
    st.session_state.setdefault('current_pdf_path', None)
    st.session_state.setdefault('current_cache_path', None)
    st.session_state.setdefault('current_index_dir', None)
    
    # Configuration and metrics
    st.session_state.setdefault('settings_configured', False)
    st.session_state.setdefault('performance_metrics', {})
    st.session_state.setdefault('processing_error', None)

def handle_file_upload(uploaded_file):
    """
    Handles the initial file upload process and path setup.
    
    Args:
        uploaded_file: The Streamlit uploaded file object.
        
    Returns:
        bool: True if a new file was detected and setup, False otherwise.
    """
    new_file_detected = (st.session_state.uploaded_filename != uploaded_file.name)
    paths_need_setting = (
        not st.session_state.current_pdf_path or 
        not isinstance(st.session_state.current_cache_path, Path) or 
        not isinstance(st.session_state.current_index_dir, Path)
    )

    if new_file_detected or paths_need_setting:
        if new_file_detected: 
            st.info(f"New file detected: {uploaded_file.name}. Initialising...")
            # Clear chat history for new file
            st.session_state.chat_history = []
        else: 
            st.info("Re-initialising file paths...")

        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.parsing_done = False
        st.session_state.indexing_done = False
        st.session_state.parsed_elements = None
        st.session_state.faiss_index = None
        st.session_state.query_engine = None
        st.session_state.settings_configured = False

        temp_file_path = UPLOAD_DIR / uploaded_file.name
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.current_pdf_path = str(temp_file_path)
        except Exception as e:
            st.error(f"Failed to save file: {e}")
            logger.error(f"Failed to save {uploaded_file.name}", exc_info=True)
            st.stop()

        st.session_state.current_cache_path = get_cache_path(uploaded_file.name)
        # To ensure it's Path
        base_index_dir = Path(UPLOAD_DIR).parent / "faiss_index_cache"  
        file_stem = Path(uploaded_file.name).stem
        sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)

        # Use '/' with Path objects
        st.session_state.current_index_dir = base_index_dir / sanitized_stem  

        logger.info(f"Set PDF path: {st.session_state.current_pdf_path}")
        logger.info(f"Set Cache path: {st.session_state.current_cache_path}")
        logger.info(f"Set Index Dir: {st.session_state.current_index_dir}")
        return True
    
    return False

def clear_cache_and_index():
    """
    Clears cached data and index for the current file.
    """
    st.session_state.parsing_done = False
    st.session_state.indexing_done = False
    st.session_state.parsed_elements = None
    st.session_state.faiss_index = None
    st.session_state.query_engine = None
    
    #Use Path objects for cache/index paths
    cache_path = st.session_state.get('current_cache_path')
    index_dir = st.session_state.get('current_index_dir')
    
    if isinstance(cache_path, Path) and cache_path.exists():
        try:
            cache_path.unlink()
            logger.info(f"Deleted cache: {cache_path}")
        except OSError as del_err:
            logger.error(f"Could not delete cache {cache_path}: {del_err}")
            
    if isinstance(index_dir, Path) and index_dir.exists():
        try:
            import shutil
            shutil.rmtree(index_dir)
            logger.info(f"Deleted index dir: {index_dir}")
        except Exception as del_idx_err:
            logger.error(f"Could not delete index {index_dir}: {del_idx_err}")
            
    st.sidebar.success("Cache & Index cleared.")

def reset_system():
    """
    Resets the entire system state.
    """
    for key in list(st.session_state.keys()):
        # Keep these parameters
        if key not in ['top_k', 'top_n']:  
            del st.session_state[key]

def run_processing_pipeline(similarity_top_k, rerank_top_n):
    """
    Executes the complete processing pipeline from parsing to query engine setup.
    
    Args:
        similarity_top_k (int): Number of nodes to retrieve.
        rerank_top_n (int): Number of nodes to keep after reranking.
        
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    # Track overall processing time
    overall_start_time = time.time()
    
    # Create status placeholders
    parsing_status, indexing_status, query_engine_status = ui.render_processing_status(
        st.session_state.parsing_done,
        st.session_state.indexing_done,
        st.session_state.query_engine,
        st.session_state.processing_error
    )
    
    # --- Step 1: Configure global settings ---
    if not st.session_state.settings_configured:
        setup_global_settings(VOYAGE_API_KEY, ANTHROPIC_API_KEY)
        st.session_state.settings_configured = True
    
    # --- Step 2: Parsing ---
    if not st.session_state.parsing_done:
        with parsing_status.container(), st.spinner("Parsing PDF..."):
            parse_start_time = time.time()
            logger.info(f"Checking cache: {st.session_state.current_cache_path.name}")
            st.session_state.parsed_elements = load_parsed_elements(st.session_state.current_cache_path)
            
            if st.session_state.parsed_elements is None:
                logger.info("Cache miss/invalid. Starting PDF parsing...")
                parser = PDFParser()
                parse_progress = st.progress(0.0, text="Starting parsing...")
                parse_start_time = time.time()
                st.session_state.parsed_elements = parser.parse_pdf(st.session_state.current_pdf_path, parse_progress)
                parse_end_time = time.time()
                parse_duration = parse_end_time - parse_start_time
                
                if st.session_state.parsed_elements:
                    st.success(f"Parsing complete ({len(st.session_state.parsed_elements)} elements) [{parse_duration:.2f}s]")
                    save_parsed_elements(st.session_state.parsed_elements, st.session_state.current_cache_path)
                    st.session_state.parsing_done = True
                    
                    # Store performance metrics
                    st.session_state.performance_metrics['parsing_time'] = parse_duration
                    st.session_state.performance_metrics['elements_count'] = len(st.session_state.parsed_elements)
                else:
                    st.error("Parsing failed.")
                    return False
            else: 
                parse_end_time = time.time()
                parse_duration = parse_end_time - parse_start_time
                st.success(f"Elements parsés chargés depuis le cache:({len(st.session_state.parsed_elements)}).")
                st.session_state.parsing_done = True
                
                # Store performance metrics
                st.session_state.performance_metrics['parsing_time'] = parse_duration
                st.session_state.performance_metrics['elements_count'] = len(st.session_state.parsed_elements)

    # --- Step 3: Node Creation & Indexing ---
    if st.session_state.parsing_done and not st.session_state.indexing_done:
        with indexing_status.container(), st.spinner("Création des Nodes-> Embedding ->Indexation..."):
            index_start_time = time.time()
            logger.info(f"Checking index at: {st.session_state.current_index_dir.name}")
            st.session_state.faiss_index = load_faiss_index(st.session_state.current_index_dir)
            
            if st.session_state.faiss_index is None:
                logger.info("Index not found. Creating Nodes and FAISS index...")
                if st.session_state.parsed_elements:
                    # Add node creation timing
                    node_start_time = time.time()
                    nodes = create_llama_nodes(st.session_state.parsed_elements)
                    node_duration = time.time() - node_start_time
                    
                    if nodes:
                        st.session_state.performance_metrics['nodes_count'] = len(nodes)
                        st.session_state.performance_metrics['node_creation_time'] = node_duration
                        
                        # Show nodes created message
                        st.info(f" {len(nodes)} nodes crées en {node_duration:.2f}s. Indexation...")
                        
                        index, indexing_time = build_faiss_index(nodes, st.session_state.current_index_dir, USE_GPU)
                        if index and indexing_time is not None: 
                            st.session_state.faiss_index = index
                            st.session_state.indexing_done = True
                            st.success(f"FAISS index created [{indexing_time:.2f}s].")
                            
                            # Store performance metrics
                            st.session_state.performance_metrics['indexing_time'] = indexing_time
                        else:
                            st.error("Index creation failed.")
                            return False
                    else:
                        st.error("Node creation failed.")
                        return False
                else:
                    st.error("Cannot index: No parsed elements.")
                    return False
            else: 
                index_end_time = time.time()
                index_duration = index_end_time - index_start_time
                st.success("FAISS index loaded from cache.")
                st.session_state.indexing_done = True
                
                # Store performance metrics
                st.session_state.performance_metrics['indexing_time'] = index_duration

    # --- Step 4: Setup Query Engine ---
    if st.session_state.indexing_done and st.session_state.query_engine is None:
        with query_engine_status.container(), st.spinner("Initialisation du moteur de requête..."):
            qe_start_time = time.time()
            st.session_state.query_engine = setup_query_engine(
                st.session_state.faiss_index, 
                VOYAGE_API_KEY, 
                top_k=similarity_top_k, 
                rerank_top_n=rerank_top_n
            )
            qe_duration = time.time() - qe_start_time
            
            if st.session_state.query_engine: 
                st.success("Query Engine ready.")
                # Store performance metrics
                st.session_state.performance_metrics['query_engine_setup_time'] = qe_duration
            else:
                st.error("Query Engine setup failed.")
                return False
    
    # Calculate and store total processing time
    overall_duration = time.time() - overall_start_time
    st.session_state.performance_metrics['total_processing_time'] = overall_duration
    
    return True

def process_query(user_query: str, top_k: int, top_n: int):
    """
    Processes a user query and updates the chat history with the response.
    
    Args:
        user_query (str): The user's question.
        top_k (int): Number of nodes to retrieve.
        top_n (int): Number of nodes to keep after reranking.
    """
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Process the query
    try:
        # Reconfigure engine if sliders changed
        if (st.session_state.query_engine._retriever.similarity_top_k != top_k or
            st.session_state.query_engine._node_postprocessors[0].top_n != top_n):
            with st.spinner("Reconfiguration du moteur de requête..."):
                logger.info(f"Paramètres modifiés. Reconfiguration du moteur de requête avec: K={top_k}, N={top_n}")
                st.session_state.query_engine = setup_query_engine(
                    st.session_state.faiss_index, 
                    VOYAGE_API_KEY, 
                    top_k=top_k, 
                    #top_n=top_n
                    rerank_top_n=top_n ###
                )
            if not st.session_state.query_engine: 
                st.error("Failed to update query engine.")
                return
        
        with st.spinner(f"Traitement: ..........."):
            query_start_time = time.time()
            
            # Generate response
            response_obj = st.session_state.query_engine.query(user_query)
            
            if response_obj:
                final_response_str = response_obj.response
                source_nodes = response_obj.source_nodes
                
                query_time = time.time() - query_start_time
                
                # Add response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"{final_response_str}\n\n*Response time: {query_time:.2f}s*"
                })
                
                # Format and add sources to chat history as system message
                if source_nodes:
                    # Sort sources by score for better presentation
                    source_nodes.sort(key=lambda n: n.score if n.score is not None else 0.0, reverse=True)
                    
                    # Build HTML for sources section
                    sources_html = f"""<div style='background-color: rgba(128, 128, 128, 0.1); padding: 10px; 
                                    border-radius: 5px; margin-top: 5px; margin-bottom: 15px; border: 1px solid rgba(128, 128, 128, 0.2);'>
                                    <p><strong>📚 Sources ({len(source_nodes)}):</strong></p>"""
                    
                    # Add each source with its metadata and snippet
                    for i, node in enumerate(source_nodes):
                        page = node.metadata.get('page_number', '?')
                        elem_type = node.metadata.get('original_element_type', 'Unknown')
                        score = node.score if node.score is not None else 0.0
                        text_snippet = node.text[:250].strip().replace("\n", " ") + "..."
                        
                        sources_html += f"""<div style='margin-bottom: 8px; padding: 5px; background-color: rgba(200, 200, 200, 0.1);'>
                                        <p><strong>Source {i+1}</strong> (Page {page}, {elem_type}, Score: {score:.4f})<br>
                                        <small>{text_snippet}</small></p></div>"""
                    
                    sources_html += "</div>"
                    
                    # Add sources as a system message in chat history
                    st.session_state.chat_history.append({
                        "role": "system",  # Special role for sources
                        "content": sources_html
                    })
            else:
                st.error("No response generated. Please try another query.")
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.error(error_msg)
        logger.error(f"Query processing error: {traceback.format_exc()}")
        st.session_state.processing_error = error_msg
        
        # Add error message to chat history
        st.session_state.chat_history.append({
            "role": "system", 
            "content": f"""<div style='background-color: rgba(255, 0, 0, 0.1); padding: 10px; 
                        border-radius: 5px; margin-bottom: 10px; border: 1px solid rgba(255, 0, 0, 0.3);'>
                        <p><strong>❌ Error:</strong> {error_msg}</p>
                        </div>"""
        })

def run_streamlit_app():
    """
    Main function to run the Streamlit application.
    """
    # Configure the page
    ui.setup_page_config()
    
    # Initialize session state
    init_session_state()
    
    # Render the header
    ui.render_header()
    
    # Display current file name if available
    if 'uploaded_filename' in st.session_state and st.session_state.uploaded_filename:
        ui.display_document_info(st.session_state.uploaded_filename)
    
    # Display metrics if available
    if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
        ui.render_metrics(st.session_state.performance_metrics)
    
    # Setup sidebar and get user inputs
    uploaded_file, similarity_top_k, rerank_top_n, force_reprocess, clear_chat, reset_system_btn = ui.setup_sidebar()
    
    # Handle clear chat button
    if clear_chat:
        st.session_state.chat_history = []
        logger.info("Chat history cleared by user.")
        st.rerun()
    
    # Handle reset system button
    if reset_system_btn:
        reset_system()
        st.rerun()
    
    # Handle force reprocess button
    if force_reprocess:
        clear_cache_and_index()
        st.rerun()
    
    # Handle file upload and processing
    if uploaded_file is not None:
        # Handle new file upload or reinitialise paths
        if handle_file_upload(uploaded_file):
            st.rerun()
            
        st.sidebar.success(f"Document: {st.session_state.uploaded_filename}")
        
        # Display processing button if pipeline is not ready
        process_button_placeholder = st.empty()
        
        if not st.session_state.query_engine:  # Show button if engine isn't ready
            if process_button_placeholder.button("Démarrez", key="process_button_key", type="primary"):
                if run_processing_pipeline(similarity_top_k, rerank_top_n):
                    st.rerun()
                    
        # Chat interface
        if st.session_state.query_engine:
            process_button_placeholder.empty()  # Hide processing button
            
            # Render chat interface
            _, user_query, send_button = ui.render_chat_interface(st.session_state.chat_history)

            #Reset the flag to clear input field after sending and render the interface
            if "submit_pressed" in st.session_state and st.session_state.submit_pressed:
                st.session_state.submit_pressed = False
            
            # Input handling
            if send_button and user_query:
                # Set flag to clear input field after sending
                st.session_state.submit_pressed = True
                
                # Process the query
                process_query(user_query, similarity_top_k, rerank_top_n)
                
                # Rerun to update the chat display
                st.rerun()
    else:
        # Show welcome screen when no document is loaded
        ui.display_welcome_screen()

# Check for API keys and required dependencies
def check_prerequisites():
    """
    Checks if all prerequisites are met before running the application.
    """
    if not VOYAGE_API_KEY or not ANTHROPIC_API_KEY:
        st.error("FATAL ERROR: API keys missing. Please configure VOYAGE_API_KEY and ANTHROPIC_API_KEY.")
        logger.critical("API Keys missing. Application cannot start.")
        return False
    elif not FAISS_AVAILABLE:
         st.error("FATAL ERROR: FAISS library not found. Please install `faiss-cpu` or `faiss-gpu`.")
         logger.critical("FAISS library missing. Application cannot start.")
         return False
    return True

# === Entry Point ===
if __name__ == "__main__":
    if check_prerequisites():
        run_streamlit_app()
