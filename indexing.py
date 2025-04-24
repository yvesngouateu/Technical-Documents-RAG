"""
Indexing Module for Temelion RAG Demo
====================================

This module handles the conversion of parsed PDF elements into vector embeddings
and manages the FAISS index for efficient similarity search.

Key functionalities include:
- Converting parsed elements to LlamaIndex TextNodes
- Setting up global LlamaIndex settings (embedding model, LLM)
- Building and persisting FAISS vector indices
- Loading persisted FAISS indices


"""

import time
import logging
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm  # For terminal progress bars

# --- LlamaIndex Core Components ---
from llama_index.core import (
    VectorStoreIndex, StorageContext, Document, Settings
)
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

# --- Vector Store Integration ---
try:
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss
except ImportError:
    pass  # The application logic will handle this in config.py

# --- Embedding Model Integration ---
from llama_index.embeddings.voyageai import VoyageEmbedding

# --- LLM Integration ---
from llama_index.llms.anthropic import Anthropic

# Import configuration
from config import (
    VOYAGE_EMBEDDING_MODEL, ANTHROPIC_LLM_MODEL, 
    FAISS_INDEX_DIM, FAISS_AVAILABLE, USE_GPU
)

# Get module-level logger
logger = logging.getLogger(__name__)

def create_llama_nodes(parsed_elements: List[Dict[str, Any]]) -> List[TextNode]:
    """
    Converts parsed elements into LlamaIndex TextNode objects using SentenceSplitter.

    Args:
        parsed_elements (List[Dict[str, Any]]): List of dictionaries from PDFParser.

    Returns:
        List[TextNode]: A list of LlamaIndex TextNode objects.
    """
    #Transforms parsed elements into chunked TextNodes for indexing.
    all_nodes: List[TextNode] = []
    text_node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    #Apply threshold for large tables
    table_chunk_size_limit = 2048  
   
    print(f"\nCreating LlamaIndex Nodes from {len(parsed_elements)} elements...")
    logger.info(f"\nCreating LlamaIndex Nodes from {len(parsed_elements)} elements...")
    
    # Use tqdm for terminal progress tracking
    element_iterator = tqdm(parsed_elements, desc="Creating Nodes")
    # Iterate over parsed elements
    # Use tqdm for terminal progress tracking
    # element_iterator = tqdm(parsed_elements, desc="Creating Nodes")

    for i, element in enumerate(element_iterator):
        content = element.get("text_content", "")
        if not content: 
            continue
        # Skip empty content
        metadata = element.get("metadata", {}).copy()
        # Copy metadata to avoid modifying the original
        metadata["original_element_type"] = element.get("element_type", "Unknown")

        # Handle large tables - simple fallback to text chunking
        if metadata["original_element_type"] == "Table":
            tokenizer = getattr(Settings, 'tokenizer', len)
            # Check if the content exceeds the chunk size limit
            # If so, chunk it as text
            if len(tokenizer(content)) > table_chunk_size_limit:
                 logger.warning(f"Large table (page {metadata.get('page_number')}) chunked as text.")
                 # Update metadata
                 metadata["original_element_type"] = "Table (Chunked)"  
        # Otherwise, use the original element type
        temp_doc = Document(text=content, metadata=metadata)
        # Create a temporary document for chunking
        # Use SentenceSplitter to chunk the text
        try:
            derived_nodes = text_node_parser.get_nodes_from_documents([temp_doc])
            all_nodes.extend(derived_nodes)
        except Exception as node_e:
            logger.error(f"Error creating nodes for element {i} (Page {metadata.get('page_number')}): {node_e}", exc_info=True)

    print(f"Created {len(all_nodes)} TextNode objects (chunks).")
    logger.info(f"Created {len(all_nodes)} TextNode objects (chunks).")
    return all_nodes


def setup_global_settings(voyage_api_key: str, anthropic_api_key: str) -> None:
    """
    Configures global LlamaIndex settings for embedding model and LLM.

    Args:
        voyage_api_key (str): Voyage AI API key.
        anthropic_api_key (str): Anthropic API key.
    """
    # Configures the primary AI models used by LlamaIndex.
    print("\nConfiguring global LlamaIndex settings...")
    logger.info("\nConfiguring global LlamaIndex settings...")
    # Set up the global settings for LlamaIndex
    try:
        if not voyage_api_key: 
            raise ValueError("Voyage API key is missing.")
        if not anthropic_api_key: 
            raise ValueError("Anthropic API key is missing.")
        # Set up the embedding model
        # Set up the LLM
        # Check if the embedding model is already set
        Settings.embed_model = VoyageEmbedding(
            model_name=VOYAGE_EMBEDDING_MODEL, 
            voyage_api_key=voyage_api_key, 
            truncation=True
        )
        print(f"  Embed Model set to: {VOYAGE_EMBEDDING_MODEL}")
        logger.info(f"  Embed Model set: {VOYAGE_EMBEDDING_MODEL}")
        
        Settings.llm = Anthropic(
            model=ANTHROPIC_LLM_MODEL, 
            api_key=anthropic_api_key, 
            max_tokens=8192, 
            timeout=120.0
        )
        print(f"  LLM set to: {ANTHROPIC_LLM_MODEL}")
        logger.info(f"  LLM set: {ANTHROPIC_LLM_MODEL}")
    except Exception as e:
        st.error(f"Error configuring LlamaIndex Settings: {e}")
        logger.critical("Failed LlamaIndex Settings", exc_info=True)
        st.stop()

def build_faiss_index(nodes: List[TextNode], persist_path: Path, use_gpu: bool) -> Tuple[Optional[VectorStoreIndex], Optional[float]]:
    """
    Builds and persists a FAISS VectorStoreIndex from LlamaIndex nodes.

    Args:
        nodes (List[TextNode]): The nodes to index.
        persist_path (Path): Directory path to save the FAISS index.
        use_gpu (bool): Flag to attempt using GPU for FAISS.

    Returns:
        Tuple[Optional[VectorStoreIndex], Optional[float]]: The index and indexing time, or (None, None) on failure.
    """
    # Create or update the vector search index.
    # This function builds a FAISS index from the provided nodes and persists it to disk.
    if not FAISS_AVAILABLE: 
        st.error("FAISS not installed.")
        return None, None
    if not nodes: 
        logger.error("Cannot build index: No nodes.")
        st.error("No content nodes for index.")
        return None, None
    
    print(f"\nBuilding FAISS index at {persist_path} (GPU requested: {use_gpu})...")
    logger.info(f"\nBuilding FAISS index at {persist_path} (GPU requested: {use_gpu})...")
    
    start_time = time.time()
    index = None
    gpu_used = False
    # Check if the FAISS index already exists
    # If it does, load it instead of creating a new one
    try:
        if not hasattr(Settings, 'embed_model') or not Settings.embed_model: 
            raise ValueError("Embed model not set.")
        #Use configured dimension
        emb_dim = FAISS_INDEX_DIM  

        faiss_cpu_index = faiss.IndexFlatL2(emb_dim)
        faiss_instance = faiss_cpu_index
        # Check if the index is already built
        # If it is, load it
        # Otherwise, create a new one

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_cpu_index)
                gpu_used = True
                print("  FAISS index using GPU.")
                logger.info("  FAISS index using GPU.")
            except Exception as gpu_e: 
                print(f"  Could not use GPU for FAISS ({type(gpu_e).__name__}). Using CPU.")
                logger.warning(f"  Could not use GPU for FAISS ({type(gpu_e).__name__}). Using CPU.")
        # If GPU is not available, use CPU
        vector_store = FaissVectorStore(faiss_index=faiss_instance)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Check if the index already exists
        # If it does, load it
        print(f" Creation des embeddings pour {len(nodes)} nodes (avec {VOYAGE_EMBEDDING_MODEL})...")
        logger.info(f"Creation des embeddings pour: {len(nodes)} nodes...")
        
        #Create a progress container in the UI
        embedding_progress = st.empty()
        embedding_text = embedding_progress.text(f"Début du process d'embedding pour ; {len(nodes)} chunks...")
        
        #Track embedding time separately
        with st.spinner(f"Creation des embeddings & indexation avec: ({len(nodes)} chunks)..."):
             start_emb = time.time()
             # emporarily disable LLM during indexing just in case
             llm_backup = getattr(Settings, 'llm', None)
             Settings.llm = None
             
             #Update progress periodically during embedding
             embedding_text.text(f"Creation des embeddings avec :{len(nodes)} chunks... Veuillez patienter, cela prendra quelques minutes.")
             
             # Use show_progress=True to get the tqdm progress bar in the terminal
             index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
             # Restore LLM
             Settings.llm = llm_backup  
             
             emb_time = time.time() - start_emb
             embedding_text.success(f"Embedding Terminé! {len(nodes)} chunks traités en: {emb_time:.2f}s")
             
            ## Log the embedding time
             print(f"    Durée Embedding/Indexation: {emb_time:.2f}s")
             logger.info(f"    Durée Embedding/Indexation: {emb_time:.2f}s")
        # Persist the index to disk
        # Create the persist directory if it doesn't exist
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_path))
        total_time = time.time() - start_time
        
        print(f"FAISS index built successfully in {total_time:.2f} seconds (GPU used: {gpu_used}).")
        logger.info(f"FAISS index built/persisted ({len(nodes)} nodes) in {total_time:.2f}s (GPU={gpu_used}) at {persist_path}")
        
        return index, total_time
    except Exception as e: 
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        st.error(f"Index build error: {e}")
        return None, None

def load_faiss_index(persist_path: Path) -> Optional[VectorStoreIndex]:
    """
    Loads a persisted FAISS VectorStoreIndex.

    Args:
        persist_path (Path): Directory path where the index is saved.

    Returns:
        Optional[VectorStoreIndex]: The loaded index, or None if loading fails.
    """
    # Load the existing index from disk.
    if not FAISS_AVAILABLE: 
        st.error("FAISS not installed.")
        return None
    # Check if the persist path exists
    # If not, return None
    print(f"Attempting to load FAISS index from {persist_path}...")
    logger.info(f"Attempting to load FAISS index from {persist_path}...")
    
    if not persist_path.exists() or not (persist_path / "vector_store.faiss").exists():
         logger.info("Index directory or file not found.")
         print("Index directory or file not found.")
         return None
    # Check if the FAISS index already exists
    # If it does, load it instead of creating a new one
    # If it doesn't, return None   
    try:
        if not hasattr(Settings, 'embed_model') or not Settings.embed_model: 
            raise ValueError("Embed model must be set before loading.")
            
        vector_store = FaissVectorStore.from_persist_dir(str(persist_path))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir=str(persist_path)
        )
        
        llm_backup = getattr(Settings, 'llm', None)
        Settings.llm = None  # Avoid LLM dependency
        
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
        
        Settings.llm = llm_backup  # Restore
        
        print("FAISS index loaded successfully.")
        logger.info("FAISS index loaded successfully.")
        
        return index
    except Exception as e: 
        logger.error(f"Error loading FAISS index: {e}", exc_info=True)
        st.warning(f"Could not load index. Rebuilding might be needed.")
        return None
