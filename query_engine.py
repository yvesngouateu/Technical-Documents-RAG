"""
Query Engine Module for Temelion RAG Demo
========================================

This module configures the query engine used for answering user questions
based on indexed PDF content. It integrates retrieval, reranking, and LLM generation
with customised prompt templates for different response formats.

Key components:
- Vector retriever setup with top-k parameter
- Reranker configuration for result refinement
- Custom prompt templates for different response formats
- Complete query engine setup with all components integrated

Usage:
    from query_engine import setup_query_engine
    
    query_engine = setup_query_engine(index, voyage_api_key, top_k=10, rerank_top_n=3)
    response = query_engine.query("What does the document say about...")
"""

import logging
import streamlit as st
from typing import Optional

# --- LlamaIndex Core Components ---
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex

# --- Reranker Integration ---
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

# Import configuration
from config import VOYAGE_RERANK_MODEL

# Get module-level logger
logger = logging.getLogger(__name__)

def setup_query_engine(
    index: VectorStoreIndex, 
    voyage_api_key: str, 
    top_k: int, 
    rerank_top_n: int
) -> Optional[RetrieverQueryEngine]:
    """
    Sets up the RetrieverQueryEngine with specific retriever and reranker settings.
    Includes custom prompt templates for formatting responses based on query keywords.

    Args:
        index (VectorStoreIndex): The loaded VectorStoreIndex.
        voyage_api_key (str): Voyage AI API key for the reranker.
        top_k (int): Number of nodes for the retriever to fetch initially.
        rerank_top_n (int): Number of nodes the reranker should return.

    Returns:
        Optional[RetrieverQueryEngine]: Configured query engine or None on failure.
    """
    # Configure the query engine with retrieval and re-ranking steps.
    if not index: 
        logger.error("Cannot setup query engine: Index missing.")
        return None
    
    print(f"\nConfiguring Query Engine (K={top_k}, N={rerank_top_n})...")
    logger.info(f"\nConfiguring Query Engine (K={top_k}, N={rerank_top_n})...")
    
    try:
        # Set up the retriever
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        print(f"  Retriever configured to fetch top {top_k} similar nodes.")
        logger.info(f"  Retriever set for top {top_k} nodes.")
        
        # Set up the reranker
        reranker = VoyageAIRerank(
            model=VOYAGE_RERANK_MODEL, 
            top_n=rerank_top_n, 
            api_key=voyage_api_key, 
            truncation=True
        )
        print(f"  Reranker configured with model '{VOYAGE_RERANK_MODEL}' to keep top {rerank_top_n} nodes.")
        logger.info(f"  Reranker set for model '{VOYAGE_RERANK_MODEL}', top {rerank_top_n} nodes.")

        # Check if LLM is configured
        if not hasattr(Settings, 'llm') or not Settings.llm: 
            raise ValueError("LLM must be configured in Settings.")

        # Set up custom prompt template for different response formats
        custom_qa_template_str = (
            "You are a knowledgeable assistant specialized in technical document analysis.\n"
            "By default, strictly answer using the same language than the prompt, unless you are explicit tasked to do otherwise.\n"
            "Please analyze the context information and answer the query thoughtfully.\n\n"
            "Context information is provided below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"
            "Query: {query_str}\n\n"
            "If the query contains 'format json', unless you are explicitly asked to do so (for example to format the entire final response strictly"
            "as a single JSON object, without any introductory text or explanations,)"
            "always format the entire final response in strict accordance with the following sequence:"
            "an introduction -> followed by an object in JSON format -> followed by any explanations.\n"

            "If the query contains 'format matrice' or 'format matrix', unless you are explicitly asked to do so (for example to format the entire final response strictly "
            "as a Markdown table (format matrix, format matrice, ), without any introductory text or explanations,),"
            "always format the entire final response in strict accordance with the following sequence:"
            "an introduction -> followed by Markdown table (matrix) -> followed by any explanations.\n"

            "If the query contains 'a numbered list', unless you are explicitly asked to do so (for example to format the entire final response strictly "
            "as a numbered list, with each point on a new line, without an introductory text),"
            "always format the entire final response in strict accordance with the following sequence:"
            "an introduction -> followed by a numbered list, with each point on a new line -> followed by any explanations.\n\n"
            "Answer: "
        )

        # Create the custom prompt template
        custom_prompt = PromptTemplate(template=custom_qa_template_str)
        
        # Create the query engine with custom prompt
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, 
            node_postprocessors=[reranker], 
            llm=Settings.llm,
            text_qa_template=custom_prompt
        )
        
        print("Query Engine configured successfully with custom prompts.")
        logger.info("Query Engine configured successfully with custom prompts.")
        
        return query_engine
    except Exception as e: 
        logger.error(f"Error setting up query engine: {e}", exc_info=True)
        st.error(f"Query Engine setup error: {e}")
        return None
