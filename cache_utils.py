"""
Cache Utilities Module for Technical Document RAG Demo
===========================================

This module provides functions for saving and loading parsed document data,
enabling persistence between application runs and avoiding redundant processing.

The module supports:
- Generating consistent cache paths for documents
- Saving parsed document elements to disk
- Loading cached document elements from disk with error handling

Usage:
    from cache_utils import get_cache_path, save_parsed_elements, load_parsed_elements
    
    cache_path = get_cache_path("document.pdf")
    save_parsed_elements(parsed_data, cache_path)
    data = load_parsed_elements(cache_path)
"""

import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import configuration
from config import CACHE_DIR

# Get module-level logger
logger = logging.getLogger(__name__)

def get_cache_path(uploaded_file_name: str) -> Path:
    """
    Generates the specific cache file path for parsed elements of a given PDF file.

    Args:
        uploaded_file_name (str): The name of the uploaded PDF file.

    Returns:
        Path: The full path to the corresponding pickle cache file.
    """
    #Creates a unique cache file path based on the PDF name.
    file_stem = Path(uploaded_file_name).stem
    #Sanitise stem to remove characters invalid for filenames/paths
    sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)
    return CACHE_DIR / f"{sanitized_stem}_parsed_cache.pkl"

def save_parsed_elements(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Saves the list of parsed elements to a pickle file.

    Args:
        data (List[Dict[str, Any]]): The list of parsed element dictionaries.
        file_path (Path): The path to the file where data will be saved.
    """
    #Saves parsed document data to a cache file.
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as f: 
            pickle.dump(data, f)
        logger.info(f"Parsed elements successfully saved to {file_path}")
        print(f"Parsed elements successfully saved to {file_path}")
    except Exception as e: 
        logger.error(f"Error saving parsed elements to {file_path}: {e}", exc_info=True)

def load_parsed_elements(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Loads the list of parsed elements from a pickle file.

    Args:
        file_path (Path): The path to the pickle file.

    Returns:
        Optional[List[Dict[str, Any]]]: Loaded data or None on failure/not found.
    """
    #Loads cached parsed data if available.
    if file_path.exists():
        try:
            with file_path.open('rb') as f: 
                data = pickle.load(f)
            logger.info(f"Parsed elements successfully loaded from {file_path}")
            print(f"Parsed elements successfully loaded from {file_path}")
            return data
        except Exception as e:
             logger.error(f"Error loading cache file {file_path}: {e}. Re-parsing needed.", exc_info=True)
             try:
                #Delete corrupt file
                file_path.unlink()  
             except OSError as del_err: 
                logger.error(f"Could not delete cache file {file_path}: {del_err}")
             return None
    else:
        logger.info(f"Cache file not found at {file_path}. Need to parse.")
        print(f"Cache file not found at {file_path}. Need to parse.")
        return None
