"""
Configuration Module for Temelion RAG Demo
==========================================

This module centralises all configuration settings for the RAG application,
including model parameters, API keys, path configurations, and processing thresholds.
It serves as a single source of truth for application settings, making maintenance
and updates more manageable.

The configuration is organised into logical sections:
- Path configurations for file storage and assets
- API keys and model settings
- PDF parsing parameters
- Interface elements
- System settings

Usage:
    from config import UPLOAD_DIR, VOYAGE_API_KEY, etc.
"""

import os
import logging
import warnings
import torch
from pathlib import Path

# === Logging Configuration ===
log_level = logging.WARNING if os.getenv("STREAMLIT_RUNNING") else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Define logger for the current module
logging.getLogger("camelot").setLevel(logging.ERROR)  # Silence Camelot's own logs
warnings.filterwarnings("ignore", message=".*No Unicode mapping.*")  # Filter specific PyMuPDF warning

# === Path Configurations ===
# --- Directory Paths ---
UPLOAD_DIR = Path("./uploaded_pdfs")
CACHE_DIR = Path("./cache")
FAISS_INDEX_DIR = Path("./faiss_index_cache")

# --- Image Paths ---
MAIN_LOGO_PATH = Path("./assets/logo_temelion.png")
SIDEBAR_ICON_PATH = Path("./assets/ai_building_hand.png")
MAIN_PATH = Path("./assets/ai_building_large.png")

# === API Keys & Model Settings ===
# --- API Keys (Replace with provided keys) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  

# --- Model Names & Parameters ---
VOYAGE_EMBEDDING_MODEL = "voyage-3-large"
VOYAGE_RERANK_MODEL = "rerank-2"
ANTHROPIC_LLM_MODEL = "claude-3-7-sonnet-20250219"
FAISS_INDEX_DIM = 1024  # Known dimension for voyage-3-large

# === PDF Parsing Configuration ===
# --- OCR & Table Extraction Parameters ---
OCR_THRESHOLD_CHARS = 50  # Minimum character count to consider OCR text useful
MIN_TABLE_ROWS = 3
MIN_TABLE_COLS = 2
TABLE_KEYWORDS = ["tableau", "matrice", "grille"] # Additional keywords can be added like: "figure", "annexe", "liste"
TABLE_HEURISTIC_MIN_LINES = 5
TABLE_HEURISTIC_NUMERIC_RATIO = 0.4

# === Hardware Acceleration Detection ===
# Check for FAISS installation and GPU availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Determine if GPU should be used for FAISS
USE_GPU = FAISS_AVAILABLE and torch.cuda.is_available()

# Log hardware detection outcomes
gpu_message = "GPU detected by PyTorch. FAISS will attempt to use it." if USE_GPU else "No GPU detected by PyTorch. FAISS will use CPU."
logger.info(f"GPU available for PyTorch: {torch.cuda.is_available()}")
logger.info(f"FAISS library available: {FAISS_AVAILABLE}")
logger.info(f"Attempting to use GPU for FAISS: {USE_GPU}")

# Check for Camelot (optional dependency for table extraction)
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# === Directory Initialization ===
# Ensure all necessary directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)
