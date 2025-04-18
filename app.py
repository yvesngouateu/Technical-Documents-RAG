# === Imports ===
import streamlit as st
import os
import re
import time
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple, Optional
import traceback  # For detailed error logging
import warnings
import logging
import torch  # For GPU check
from tqdm import tqdm  # For terminal progress bars

# --- Parsing Dependencies ---
import fitz  # PyMuPDF
import pytesseract  # For OCR
from PIL import Image
import io  # For image conversion
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    # Logger will be defined later
import pandas as pd

# --- LlamaIndex Core Components ---
from llama_index.core import (
    VectorStoreIndex, StorageContext, Document, Settings
)
from llama_index.core.schema import TextNode, NodeRelationship, NodeWithScore, QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate

# --- Vector Store Integration ---
try:
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # The application logic will handle this later if needed.

# --- Embedding Model Integration ---
from llama_index.embeddings.voyageai import VoyageEmbedding

# --- Reranker Integration ---
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

# --- LLM Integration ---
from llama_index.llms.anthropic import Anthropic

# === Configuration ===

st.set_page_config(page_title="Temelion RAG Demo", layout="wide")
# Configure logging (place after imports, before first use)
log_level = logging.WARNING if os.getenv("STREAMLIT_RUNNING") else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Define logger for the current module
logging.getLogger("camelot").setLevel(logging.ERROR)  # Silence Camelot's own logs
warnings.filterwarnings("ignore", message=".*No Unicode mapping.*")  # Filter specific PyMuPDF warning


# --- PDF Parsing Config ---
OCR_THRESHOLD_CHARS = 50  # Minimum character count to consider OCR text useful
MIN_TABLE_ROWS = 3
MIN_TABLE_COLS = 2
TABLE_KEYWORDS = ["tableau", "matrice", "grille"] #, "figure", "annexe", "liste"
TABLE_HEURISTIC_MIN_LINES = 5
TABLE_HEURISTIC_NUMERIC_RATIO = 0.4

# --- Image Paths ---
#IMAGES_DIR = Path("./assets")
MAIN_LOGO_PATH = Path("./assets/logo_temelion.png")
SIDEBAR_ICON_PATH = Path("./assets/ai_building_hand.png")
MAIN_PATH = Path("./assets/ai_building_large.png")
# Create images directory if it doesn't exist
#IMAGES_DIR.mkdir(exist_ok=True)

# --- API Keys & Models # Replace with your actual key if needed ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VOYAGE_EMBEDDING_MODEL = "voyage-3-large"
VOYAGE_RERANK_MODEL = "rerank-2"
ANTHROPIC_LLM_MODEL = "claude-3-7-sonnet-20250219"
FAISS_INDEX_DIM = 1024  # Known dimension for voyage-3-large

# --- FAISS GPU Check ---
use_gpu = FAISS_AVAILABLE and torch.cuda.is_available()  # Check both FAISS install and GPU hardware
gpu_message = "GPU detected by PyTorch. FAISS will attempt to use it." if use_gpu else "No GPU detected by PyTorch. FAISS will use CPU."
print(gpu_message)  # Print to console
logger.info(f"GPU available for PyTorch: {torch.cuda.is_available()}")
logger.info(f"FAISS library available: {FAISS_AVAILABLE}")
logger.info(f"Attempting to use GPU for FAISS: {use_gpu}")


# --- File Paths ---
UPLOAD_DIR = Path("./uploaded_pdfs")
CACHE_DIR = Path("./cache")
FAISS_INDEX_DIR = Path("./faiss_index_cache")  # Define the base directory for FAISS indices

# Ensure all necessary directories exist at startup
UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)  # Ensure base FAISS directory exists

# === Helper Function for Cache Path ===
def get_cache_path(uploaded_file_name: str) -> Path:
    """
    Generates the specific cache file path for parsed elements of a given PDF file.

    Args:
        uploaded_file_name (str): The name of the uploaded PDF file.

    Returns:
        Path: The full path to the corresponding pickle cache file.
    """
    # Creates a unique cache file path based on the PDF name.
    file_stem = Path(uploaded_file_name).stem
    # Sanitise stem to remove characters invalid for filenames/paths
    sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)
    return CACHE_DIR / f"{sanitized_stem}_parsed_cache.pkl"

# === PDFParser Class ===
class PDFParser:
    """
    Parses PDF documents with comprehensive feature extraction:
    - Text blocks using PyMuPDF
    - Structured tables using Camelot 
    - Image-based text via OCR (pytesseract)
    
    Provides a complete extraction pipeline suitable for any PDF document type.

    Methods:
        parse_pdf: Processes the entire PDF page by page.
    """
    def __init__(self):
        """Initialises the PDFParser."""
        logger.debug("PDFParser initialised.")

    def _extract_text_from_image_ocr(self, img_bytes: bytes) -> str:
        """
        Extracts text from an image using OCR (Optical Character Recognition).

        Args:
            img_bytes (bytes): The raw image bytes to process.

        Returns:
            str: Extracted text content from the image, or empty string on failure.
        """
        try:
            image = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(image, lang='fra+eng')
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed for an image: {e}")
            return ""

    def _is_likely_table_heuristic(self, text_block: str) -> bool:
        """
        Applies heuristics to check if a text block likely contains tabular data.

        Args:
            text_block (str): Text content of a block extracted by PyMuPDF.

        Returns:
            bool: True if the block seems likely to be a table, False otherwise.
        """
        # Heuristic check for tables before trying Camelot.
        text_lower = text_block.lower().strip()
        if not text_lower: return False
        has_keyword = any(keyword in text_lower for keyword in TABLE_KEYWORDS)
        lines = text_block.strip().split('\n')
        num_lines = len(lines)
        if num_lines < TABLE_HEURISTIC_MIN_LINES: return False

        # Calculate ratio of lines starting numerically or indented
        numerical_or_indented_lines = sum(1 for line in lines if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] in ['-','*','•'] or line.startswith("   ")))
        numeric_ratio = numerical_or_indented_lines / num_lines if num_lines > 0 else 0

        # Calculate ratio of lines resembling columns (multiple spaces)
        cols_like = sum(1 for line in lines if re.search(r'\s{2,}', line.strip()))
        col_ratio = cols_like / num_lines if num_lines > 0 else 0

        # Decision logic
        if numeric_ratio > TABLE_HEURISTIC_NUMERIC_RATIO and col_ratio > 0.5: return True
        if has_keyword and col_ratio > 0.6: return True
        return False

    def _extract_specific_tables_camelot(self, file_path: str, page_number: int) -> List[Tuple[str, fitz.Rect]]:
        """
        Attempts table extraction from a specific page using Camelot.

        Args:
            file_path (str): Path to the PDF file.
            page_number (int): The 1-based page number.

        Returns:
            List[Tuple[str, fitz.Rect]]: List of (Markdown_Table, Bounding_Box) tuples.
        """
        # Extracts tables using Camelot if available.
        if not CAMELOT_AVAILABLE: return []
        validated_tables: List[Tuple[str, fitz.Rect]] = []
        try:
            tables = camelot.read_pdf(file_path, pages=str(page_number), flavor='lattice', suppress_stdout=True, line_scale=40)
            if not tables or tables.n == 0: tables = camelot.read_pdf(file_path, pages=str(page_number), flavor='stream', suppress_stdout=True)

            if tables.n > 0: logger.info(f"Camelot found {tables.n} potential tables on page {page_number}.")
            for i, table in enumerate(tables):
                try:
                    table_df = table.df
                    if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                        # Cleaning and validation
                        table_df.columns = table_df.columns.map(lambda x: str(x).replace('\n', ' ').strip())
                        table_df = table_df.map(lambda x: str(x).replace('\n', ' ').strip()) if hasattr(table_df, 'map') else table_df.applymap(lambda x: str(x).replace('\n', ' ').strip())
                        table_df.dropna(how='all', axis=1, inplace=True); table_df.dropna(how='all', axis=0, inplace=True)

                        if table_df.shape[0] >= MIN_TABLE_ROWS and table_df.shape[1] >= MIN_TABLE_COLS:
                            markdown_table = table_df.to_markdown(index=False)
                            table_title = f"Table_{i+1}"; bbox_coords = table._bbox  # Use Camelot's internal bbox
                            full_md = f"\n<<< START Table: {table_title} (Page {page_number}) >>>\n{markdown_table}\n<<< END Table >>>\n"
                            bbox_rect = fitz.Rect(bbox_coords)  # Convert to fitz.Rect
                            validated_tables.append((full_md, bbox_rect))
                            logger.info(f"  Extracted valid table {i+1} (Shape: {table_df.shape}) as Markdown.")
                        else: logger.info(f"  Table {i+1} rejected by validation (Shape: {table_df.shape}).")
                except Exception as df_e: logger.warning(f"Page {page_number}, Table {i+1}: Error processing DataFrame: {df_e}")
        except Exception as e:
             if "ghostscript" in str(e).lower(): st.sidebar.error("Ghostscript error. Table extraction may fail.", icon="❌"); logger.error("Ghostscript error for Camelot.")
             else: logger.warning(f"Camelot parsing error on page {page_number}: {type(e).__name__}")
        return validated_tables

    def parse_pdf(self, file_path: str, progress_bar: Optional[st.progress] = None) -> List[Dict[str, Any]]:
        """
        Parses the PDF document page by page, extracting text blocks, tables, and image-based text.

        Args:
            file_path (str): Path to the PDF file.
            progress_bar (Optional[st.progress]): Streamlit progress bar object to update.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing parsed elements.
        """
        # Main parsing function iterating through pages.
        extracted_elements: List[Dict[str, Any]] = []
        doc = None
        try:
            logger.info(f"Opening PDF: {os.path.basename(file_path)}")
            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"Starting parsing: {os.path.basename(file_path)} ({total_pages} pages)")

            for page_num in range(total_pages):
                page_number_actual = page_num + 1
                logger.debug(f"Processing Page {page_number_actual}/{total_pages}")
                page = doc.load_page(page_num)
                page_rect = page.rect
                table_extraction_attempted = False

                if progress_bar:
                     progress_value = (page_num) / total_pages
                     progress_bar.progress(progress_value, text=f"Analysing Page {page_number_actual}/{total_pages}...")

                # --- Extract Tables FIRST ---
                extracted_tables_data = self._extract_specific_tables_camelot(file_path, page_number_actual)
                extracted_table_bboxes = []
                for t_idx, (table_md, table_bbox) in enumerate(extracted_tables_data):
                    table_metadata = { "source_file": os.path.basename(file_path), "page_number": page_number_actual, "block_type": "table", "extraction_method": "Camelot", "bbox": [round(c, 2) for c in table_bbox] }
                    extracted_elements.append({"page_number": page_number_actual, "element_type": "Table", "text_content": table_md, "metadata": table_metadata })
                    extracted_table_bboxes.append(table_bbox)
                    table_extraction_attempted = True

                # --- Extract Text Blocks (PyMuPDF), avoiding table areas ---
                try:
                    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXT_PRESERVE_WHITESPACE)
                    if blocks and 'blocks' in blocks:
                        for block in blocks.get('blocks', []):
                            if block['type'] == 0:  # Text block
                                block_bbox = fitz.Rect(block['bbox'])
                                # Check overlap with extracted tables
                                if any(block_bbox.intersects(t_bbox) and (block_bbox & t_bbox).get_area() / block_bbox.get_area() > 0.5 for t_bbox in extracted_table_bboxes if t_bbox.is_valid and block_bbox.is_valid):
                                    # logger.debug(f"Skipping text block overlapping table on page {page_number_actual}")
                                    continue

                                block_text = "".join(span['text'] for line in block.get('lines', []) for span in line.get('spans', [])).strip()
                                if block_text:  # Extract only if block contains text
                                    metadata = { "source_file": os.path.basename(file_path), "page_number": page_number_actual, "block_type": "text", "bbox": [round(c, 2) for c in block_bbox] }
                                    extracted_elements.append({ "page_number": page_number_actual, "element_type": "TextBlock", "text_content": block_text, "metadata": metadata })
                                    
                                    # Check if this block might contain a table (heuristic check)
                                    if not table_extraction_attempted and self._is_likely_table_heuristic(block_text):
                                        tables_from_heuristic = self._extract_specific_tables_camelot(file_path, page_number_actual)
                                        table_extraction_attempted = True
                                        
                                        # Add any tables found via heuristic
                                        for t_idx, (table_md, table_bbox) in enumerate(tables_from_heuristic):
                                            t_metadata = {"source_file": os.path.basename(file_path), "page_number": page_number_actual, 
                                                         "block_type": "table", "extraction_method": "Camelot_Heuristic", 
                                                         "bbox": [round(c, 2) for c in table_bbox]}
                                            extracted_elements.append({"page_number": page_number_actual, "element_type": "Table", 
                                                                      "text_content": table_md, "metadata": t_metadata})
                except Exception as block_e: logger.error(f"Page {page_number_actual}: Error processing text blocks: {block_e}")

                # --- Extract Images and Perform OCR ---
                try:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        bbox = page.get_image_bbox(img_info).irect  # Get rectangle containing image
                        
                        # Perform OCR on the image
                        ocr_text = self._extract_text_from_image_ocr(image_bytes)
                        
                        # Only add if OCR returned meaningful text
                        if ocr_text and len(ocr_text) > OCR_THRESHOLD_CHARS:
                            image_metadata = {
                                "source_file": os.path.basename(file_path),
                                "page_number": page_number_actual,
                                "block_type": "image_ocr",
                                "image_index": img_index,
                                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                            }
                            extracted_elements.append({
                                "page_number": page_number_actual,
                                "element_type": "ImageOCR",
                                "text_content": ocr_text,
                                "metadata": image_metadata
                            })
                            logger.info(f"  Extracted OCR text from image {img_index} on page {page_number_actual} ({len(ocr_text)} chars)")
                except Exception as img_e: 
                    logger.error(f"Page {page_number_actual}: Error processing images/OCR: {img_e}")

            if progress_bar: progress_bar.progress(1.0, text="Parsing Finished!")
            logger.info(f"Parsing complete. Found {len(extracted_elements)} elements.")
            extracted_elements.sort(key=lambda x: (x.get('page_number', 0), x.get('metadata', {}).get('bbox', [0, 0, 0, 0])[1]))

        except Exception as e:
            st.error(f"Major error during PDF parsing {file_path}: {e}")
            logger.error(f"Fatal parsing error for {file_path}", exc_info=True)
            return []
        finally:
            if doc: doc.close()
        return extracted_elements

# === Functions for Saving/Loading Parsed Data ===
def save_parsed_elements(data: List[Dict[str, Any]], file_path: Path):
    """
    Saves the list of parsed elements to a pickle file.

    Args:
        data (List[Dict[str, Any]]): The list of parsed element dictionaries.
        file_path (Path): The path to the file where data will be saved.
    """
    # Saves parsed document data to a cache file.
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as f: pickle.dump(data, f)
        logger.info(f"Parsed elements successfully saved to {file_path}")
        print(f"Parsed elements successfully saved to {file_path}")
    except Exception as e: logger.error(f"Error saving parsed elements to {file_path}: {e}", exc_info=True)

def load_parsed_elements(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Loads the list of parsed elements from a pickle file.

    Args:
        file_path (Path): The path to the pickle file.

    Returns:
        Optional[List[Dict[str, Any]]]: Loaded data or None on failure/not found.
    """
    # Loads cached parsed data if available.
    if file_path.exists():
        try:
            with file_path.open('rb') as f: data = pickle.load(f)
            logger.info(f"Parsed elements successfully loaded from {file_path}")
            print(f"Parsed elements successfully loaded from {file_path}")
            return data
        except Exception as e:
             logger.error(f"Error loading cache file {file_path}: {e}. Re-parsing needed.", exc_info=True)
             try: file_path.unlink()  # Delete corrupt file
             except OSError as del_err: logger.error(f"Could not delete cache file {file_path}: {del_err}")
             return None
    else:
        logger.info(f"Cache file not found at {file_path}. Need to parse.")
        print(f"Cache file not found at {file_path}. Need to parse.")
        return None

# === LlamaIndex Functions ===
def create_llama_nodes(parsed_elements: List[Dict[str, Any]]) -> List[TextNode]:
    """
    Converts parsed elements into LlamaIndex TextNode objects using SentenceSplitter.

    Args:
        parsed_elements (List[Dict[str, Any]]): List of dictionaries from PDFParser.

    Returns:
        List[TextNode]: A list of LlamaIndex TextNode objects.
    """
    # Transforms parsed elements into chunked TextNodes for indexing.
    all_nodes: List[TextNode] = []
    text_node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    table_chunk_size_limit = 2048  # Apply threshold for large tables

    print(f"\nCreating LlamaIndex Nodes from {len(parsed_elements)} elements...")
    logger.info(f"\nCreating LlamaIndex Nodes from {len(parsed_elements)} elements...")
    
    # Use tqdm for terminal progress tracking
    element_iterator = tqdm(parsed_elements, desc="Creating Nodes")

    for i, element in enumerate(element_iterator):
        content = element.get("text_content", "")
        if not content: continue
        metadata = element.get("metadata", {}).copy()
        metadata["original_element_type"] = element.get("element_type", "Unknown")

        # Handle large tables - simple fallback to text chunking
        if metadata["original_element_type"] == "Table":
            tokenizer = getattr(Settings, 'tokenizer', len)
            if len(tokenizer(content)) > table_chunk_size_limit:
                 logger.warning(f"Large table (page {metadata.get('page_number')}) chunked as text.")
                 metadata["original_element_type"] = "Table (Chunked)"  # Update metadata

        temp_doc = Document(text=content, metadata=metadata)
        try:
            derived_nodes = text_node_parser.get_nodes_from_documents([temp_doc])
            all_nodes.extend(derived_nodes)
        except Exception as node_e:
            logger.error(f"Error creating nodes for element {i} (Page {metadata.get('page_number')}): {node_e}", exc_info=True)

    print(f"Created {len(all_nodes)} TextNode objects (chunks).")
    logger.info(f"Created {len(all_nodes)} TextNode objects (chunks).")
    return all_nodes


def setup_global_settings(voyage_api_key: str, anthropic_api_key: str):
    """
    Configures global LlamaIndex settings for embedding model and LLM.

    Args:
        voyage_api_key (str): Voyage AI API key.
        anthropic_api_key (str): Anthropic API key.
    """
    # Configures the primary AI models used by LlamaIndex.
    print("\nConfiguring global LlamaIndex settings...")
    logger.info("\nConfiguring global LlamaIndex settings...")
    try:
        if not voyage_api_key: raise ValueError("Voyage API key is missing.")
        if not anthropic_api_key: raise ValueError("Anthropic API key is missing.")
        Settings.embed_model = VoyageEmbedding(model_name=VOYAGE_EMBEDDING_MODEL, voyage_api_key=voyage_api_key, truncation=True)
        print(f"  Embed Model set to: {VOYAGE_EMBEDDING_MODEL}")
        logger.info(f"  Embed Model set: {VOYAGE_EMBEDDING_MODEL}")
        Settings.llm = Anthropic(model=ANTHROPIC_LLM_MODEL, api_key=anthropic_api_key, max_tokens=4096, timeout=120.0)
        print(f"  LLM set to: {ANTHROPIC_LLM_MODEL}")
        logger.info(f"  LLM set: {ANTHROPIC_LLM_MODEL}")
    except Exception as e:
        st.error(f"Error configuring LlamaIndex Settings: {e}")
        logger.critical("Failed LlamaIndex Settings", exc_info=True); st.stop()

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
    # Creates or updates the vector search index.
    if not FAISS_AVAILABLE: st.error("FAISS not installed."); return None, None
    if not nodes: logger.error("Cannot build index: No nodes."); st.error("No content nodes for index."); return None, None
    
    print(f"\nBuilding FAISS index at {persist_path} (GPU requested: {use_gpu})...")
    logger.info(f"\nBuilding FAISS index at {persist_path} (GPU requested: {use_gpu})...")
    
    start_time = time.time(); index = None; gpu_used = False
    try:
        if not hasattr(Settings, 'embed_model') or not Settings.embed_model: raise ValueError("Embed model not set.")
        emb_dim = FAISS_INDEX_DIM  # Use configured dimension

        faiss_cpu_index = faiss.IndexFlatL2(emb_dim)
        faiss_instance = faiss_cpu_index
        if use_gpu:
            try:
                res = faiss.StandardGpuResources(); faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_cpu_index)
                gpu_used = True
                print("  FAISS index using GPU.")
                logger.info("  FAISS index using GPU.")
            except Exception as gpu_e: 
                print(f"  Could not use GPU for FAISS ({type(gpu_e).__name__}). Using CPU.")
                logger.warning(f"  Could not use GPU for FAISS ({type(gpu_e).__name__}). Using CPU.")

        vector_store = FaissVectorStore(faiss_index=faiss_instance)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"  Generating embeddings for {len(nodes)} nodes (using {VOYAGE_EMBEDDING_MODEL})...")
        logger.info(f"Generating embeddings for {len(nodes)} nodes...")
        
        # Create a progress container in the UI
        embedding_progress = st.empty()
        embedding_text = embedding_progress.text(f"Starting embedding for {len(nodes)} chunks...")
        
        # Track embedding time separately
        with st.spinner(f"Generating embeddings & building index ({len(nodes)} chunks)..."):
             start_emb = time.time()
             # Temporarily disable LLM during indexing just in case
             llm_backup = getattr(Settings, 'llm', None); Settings.llm = None
             
             # Update progress periodically during embedding
             embedding_text.text(f"Generating embeddings for {len(nodes)} chunks... Please wait, this may take several minutes.")
             
             # Use show_progress=True to get the tqdm progress bar in the terminal
             index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
             
             Settings.llm = llm_backup  # Restore LLM
             
             emb_time = time.time() - start_emb
             embedding_text.success(f"Embedding complete! {len(nodes)} chunks processed in {emb_time:.2f}s")
             
             print(f"    Embedding/Indexing took: {emb_time:.2f}s")
             logger.info(f"    Embedding/Indexing took: {emb_time:.2f}s")

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
    # Loads the existing index from disk.
    if not FAISS_AVAILABLE: st.error("FAISS not installed."); return None
    
    print(f"Attempting to load FAISS index from {persist_path}...")
    logger.info(f"Attempting to load FAISS index from {persist_path}...")
    
    if not persist_path.exists() or not (persist_path / "vector_store.faiss").exists():
         logger.info("Index directory or file not found."); 
         print("Index directory or file not found.")
         return None
    try:
        if not hasattr(Settings, 'embed_model') or not Settings.embed_model: raise ValueError("Embed model must be set before loading.")
        vector_store = FaissVectorStore.from_persist_dir(str(persist_path))
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_path))
        llm_backup = getattr(Settings, 'llm', None); Settings.llm = None  # Avoid LLM dependency
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        Settings.llm = llm_backup  # Restore
        
        print("FAISS index loaded successfully.")
        logger.info("FAISS index loaded successfully.")
        
        return index
    except Exception as e: 
        logger.error(f"Error loading FAISS index: {e}", exc_info=True)
        st.warning(f"Could not load index. Rebuilding might be needed.")
        return None

def setup_query_engine(index: VectorStoreIndex, voyage_api_key: str, top_k: int, rerank_top_n: int) -> Optional[RetrieverQueryEngine]:
    """
    Sets up the RetrieverQueryEngine with specific retriever and reranker settings.
    Includes a simple prompt template for consistency.

    Args:
        index (VectorStoreIndex): The loaded VectorStoreIndex.
        voyage_api_key (str): Voyage AI API key for the reranker.
        top_k (int): Number of nodes for the retriever to fetch initially.
        rerank_top_n (int): Number of nodes the reranker should return.

    Returns:
        Optional[RetrieverQueryEngine]: Configured query engine or None on failure.
    """
    # Configures the query engine with retrieval and re-ranking steps.
    if not index: logger.error("Cannot setup query engine: Index missing."); return None
    
    print(f"\nConfiguring Query Engine (K={top_k}, N={rerank_top_n})...")
    logger.info(f"\nConfiguring Query Engine (K={top_k}, N={rerank_top_n})...")
    
    try:
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        print(f"  Retriever configured to fetch top {top_k} similar nodes.")
        logger.info(f"  Retriever set for top {top_k} nodes.")
        
        reranker = VoyageAIRerank(model=VOYAGE_RERANK_MODEL, top_n=rerank_top_n, api_key=voyage_api_key, truncation=True)
        print(f"  Reranker configured with model '{VOYAGE_RERANK_MODEL}' to keep top {rerank_top_n} nodes.")
        logger.info(f"  Reranker set for model '{VOYAGE_RERANK_MODEL}', top {rerank_top_n} nodes.")

        if not hasattr(Settings, 'llm') or not Settings.llm: raise ValueError("LLM must be configured in Settings.")
        
        # Create a basic prompt template without complex parameters
        custom_qa_template_str = (
            "Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Réponse: "
        )
        
        # Create the prompt template with just the template string
        custom_prompt = PromptTemplate(template=custom_qa_template_str)
        
        # Create query engine with the basic prompt template
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, 
            node_postprocessors=[reranker], 
            llm=Settings.llm,
            text_qa_template=custom_prompt
        )
        
        print("Query Engine configured successfully.")
        logger.info("Query Engine configured successfully.")
        
        return query_engine
    except Exception as e: 
        logger.error(f"Error setting up query engine: {e}", exc_info=True)
        st.error(f"Query Engine setup error: {e}")
        return None

# === Helper function for query formatting ===
def enhance_query_with_formatting(query_str: str) -> str:
    """
    Enhances a query with formatting instructions based on keywords.
    
    Args:
        query_str (str): The original query string
        
    Returns:
        str: Enhanced query with formatting instructions if needed
    """
    enhanced_query = query_str
    query_lower = query_str.lower()
    
    # Add formatting instructions when keywords are detected
    if " format json" in query_lower:
        enhanced_query += "\n\nIMPORTANT: unless you are explicitly asked to do so (for example to format the entire final response strictly "
        "as a single JSON object, without any introductory text or explanations,)"
        "always format the entire final response in strict accordance with the following sequence:"
        "an introduction -> followed by an object in JSON format -> followed by any explanations."
    elif " format matrice" in query_lower or " format matrix" in query_lower:
        enhanced_query += "\n\nIMPORTANT: unless you are explicitly asked to do so (for example to format the entire final response strictly "
        "as a Markdown table (matrix), without any introductory text or explanations.),"
        "always format the entire final response in strict accordance with the following sequence:"
        "an introduction -> followed by Markdown table (matrix) -> followed by any explanations."
    elif " format liste" in query_lower or " format list" in query_lower:
        enhanced_query += "\n\nIMPORTANT: unless you are explicitly asked to do so (for example to format the entire final response strictly "
        "as a numbered list, with each point on a new line., without an introductory text.),"
        "always format the entire final response in strict accordance with the following sequence:"
        "an introduction -> followed by a numbered list, with each point on a new line. -> followed by any explanations."
    return enhanced_query

# === Streamlit Application Logic ===
def run_streamlit_app():
    """Main function defining and running the Streamlit application interface and logic."""
    # Initialize conversation history if not already in session
    st.session_state.setdefault('chat_history', [])
    
    # Layout title with logo
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if MAIN_LOGO_PATH.exists():
            st.image(str(MAIN_LOGO_PATH), width=100)
        else:
            st.info(f"Main logo not found at {MAIN_LOGO_PATH}. Place an image there to display it here.")
            
    with title_col:
        st.title("Temelion RAG Demo - vOptimised")
    
    # Display the filename in bold uppercase if available
    if st.session_state.get('uploaded_filename'):
        st.markdown(f"**DOCUMENT: {st.session_state.get('uploaded_filename').upper()}**")
    
    # Sidebar with icon
    if SIDEBAR_ICON_PATH.exists():
        st.sidebar.image(str(SIDEBAR_ICON_PATH), width=50)
    else:
        st.sidebar.info(f"Sidebar icon not found at {SIDEBAR_ICON_PATH}. Place an image there to display it here.")
    
    st.sidebar.header("Configuration & Control")

    # --- File Upload with extension verification ---
    uploaded_file = st.sidebar.file_uploader("Upload PDF document", type="pdf", key="pdf_uploader")

    # --- Initialise Session State ---
    st.session_state.setdefault('parsing_done', False)
    st.session_state.setdefault('indexing_done', False)
    st.session_state.setdefault('parsed_elements', None)
    st.session_state.setdefault('faiss_index', None)
    st.session_state.setdefault('query_engine', None)
    st.session_state.setdefault('uploaded_filename', None)
    st.session_state.setdefault('current_pdf_path', None)
    st.session_state.setdefault('current_cache_path', None)
    st.session_state.setdefault('current_index_dir', None)
    st.session_state.setdefault('settings_configured', False)
    st.session_state.setdefault('performance_metrics', {})

    # --- Control Panel ---
    similarity_top_k = st.sidebar.slider("Nodes Retrieved (K)", min_value=3, max_value=20, value=10, step=1, key="top_k", help="Chunks retrieved before re-ranking.")
    rerank_top_n = st.sidebar.slider("Nodes Reranked (N)", min_value=1, max_value=10, value=3, step=1, key="top_n", help="Chunks kept after re-ranking.")
    
    # Processing control buttons
    force_reprocess = st.sidebar.button("Force Re-Parse & Re-Index", key="force_reindex_button", help="Clears cache and index for the current file.")
    
    # Add divider for chat controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chat Controls**")
    
    # Button to clear chat history
    clear_chat = st.sidebar.button("Clear Chat History", key="clear_chat_button", help="Clear conversation history while keeping the current document")
    if clear_chat:
        st.session_state.chat_history = []
        st.sidebar.success("Chat history cleared!")
        st.rerun()
    
    # Button to reset entire system
    reset_system = st.sidebar.button("Reset Entire System", key="reset_system_button", help="Start over with a new document")
    if reset_system:
        # Preserve only a few session variables to avoid errors
        temp_keys = {}
        # Reset session state completely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("System reset complete!")
        st.rerun()

    if force_reprocess:
        # Clears cached data and index for the current file.
        st.session_state.parsing_done = False
        st.session_state.indexing_done = False
        st.session_state.parsed_elements = None
        st.session_state.faiss_index = None
        st.session_state.query_engine = None
        # Use Path objects for cache/index paths
        cache_path = st.session_state.get('current_cache_path')
        index_dir = st.session_state.get('current_index_dir')
        if cache_path and cache_path.exists():
            try: cache_path.unlink(); logger.info(f"Deleted cache: {cache_path}")
            except OSError as del_err: logger.error(f"Could not delete cache {cache_path}: {del_err}")
        if index_dir and index_dir.exists():
            try: import shutil; shutil.rmtree(index_dir); logger.info(f"Deleted index dir: {index_dir}")
            except Exception as del_idx_err: logger.error(f"Could not delete index {index_dir}: {del_idx_err}")
        st.sidebar.success("Cache & Index cleared.")
        st.rerun()

    # --- Main Application Flow ---
    if uploaded_file is not None:
        # Verify file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension != ".pdf":
            st.error("Please provide a PDF document (.pdf extension required)")
            st.stop()

        # --- Handle New File Upload or Path Initialisation ---
        # Manages state when a new file is uploaded or session resets.
        new_file_detected = (st.session_state.uploaded_filename != uploaded_file.name)
        paths_need_setting = not st.session_state.current_pdf_path or not isinstance(st.session_state.current_cache_path, Path) or not isinstance(st.session_state.current_index_dir, Path)

        if new_file_detected or paths_need_setting:
            if new_file_detected: 
                st.info(f"New file detected: {uploaded_file.name}. Initialising...")
                # Clear chat history when loading a new file
                st.session_state.chat_history = []
            else: st.info("Re-initialising file paths...")

            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.parsing_done = False; st.session_state.indexing_done = False
            st.session_state.parsed_elements = None; st.session_state.faiss_index = None
            st.session_state.query_engine = None; st.session_state.settings_configured = False

            temp_file_path = UPLOAD_DIR / uploaded_file.name
            try:
                with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.current_pdf_path = str(temp_file_path)
            except Exception as e: st.error(f"Failed to save file: {e}"); logger.error(f"Failed to save {uploaded_file.name}", exc_info=True); st.stop()

            st.session_state.current_cache_path = get_cache_path(uploaded_file.name)
            base_index_dir = Path(FAISS_INDEX_DIR)  # Ensure it's Path
            file_stem = Path(uploaded_file.name).stem
            sanitized_stem = re.sub(r'[^\w\-_\. ]', '_', file_stem)
            st.session_state.current_index_dir = base_index_dir / sanitized_stem  # Use '/' with Path objects

            logger.info(f"Set PDF path: {st.session_state.current_pdf_path}")
            logger.info(f"Set Cache path: {st.session_state.current_cache_path}")
            logger.info(f"Set Index Dir: {st.session_state.current_index_dir}")
            st.rerun()

        st.sidebar.success(f"Document: {st.session_state.uploaded_filename}")

        # --- Configure Settings (Ensures it runs once after paths are set) ---
        if not st.session_state.settings_configured:
             setup_global_settings(VOYAGE_API_KEY, ANTHROPIC_API_KEY)
             st.session_state.settings_configured = True

        # --- Status Placeholders ---
        col1, col2, col3 = st.columns(3)
        with col1: parsing_status_placeholder = st.empty()
        with col2: indexing_status_placeholder = st.empty()
        with col3: query_engine_status_placeholder = st.empty()
        process_button_placeholder = st.empty()
        
        # --- Performance Metrics Display ---
        # Add a performance metrics section
        metrics_placeholder = st.empty()

        # --- Processing Pipeline ---
        # Controls the parsing and indexing workflow.
        if not st.session_state.query_engine:  # Show button if engine isn't ready
            if process_button_placeholder.button("Process Document (Parse & Index)", key="process_button_key"):
                # Track overall processing time
                overall_start_time = time.time()
                
                # PARSING
                if not st.session_state.parsing_done:
                    with parsing_status_placeholder.container(), st.spinner("Parsing PDF..."):
                         parse_start_time = time.time()
                         logger.info(f"Checking cache: {st.session_state.current_cache_path.name}")
                         st.session_state.parsed_elements = load_parsed_elements(st.session_state.current_cache_path)
                         if st.session_state.parsed_elements is None:
                             logger.info("Cache miss/invalid. Starting PDF parsing...")
                             parser = PDFParser(); parse_progress = st.progress(0.0, text="Starting parsing...")
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
                             else: st.error("Parsing failed."); st.stop()
                         else: 
                             parse_end_time = time.time()
                             parse_duration = parse_end_time - parse_start_time
                             st.success(f"Parsed elements loaded from cache ({len(st.session_state.parsed_elements)}).")
                             st.session_state.parsing_done = True
                             # Store performance metrics
                             st.session_state.performance_metrics['parsing_time'] = parse_duration
                             st.session_state.performance_metrics['elements_count'] = len(st.session_state.parsed_elements)

                # INDEXING
                if st.session_state.parsing_done and not st.session_state.indexing_done:
                     with indexing_status_placeholder.container(), st.spinner("Indexing..."):
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
                                     st.info(f"Created {len(nodes)} chunks in {node_duration:.2f}s. Building index...")
                                     
                                     index, indexing_time = build_faiss_index(nodes, st.session_state.current_index_dir, use_gpu)
                                     if index and indexing_time is not None: 
                                         st.session_state.faiss_index = index
                                         st.session_state.indexing_done = True
                                         st.success(f"FAISS index created [{indexing_time:.2f}s].")
                                         
                                         # Store performance metrics
                                         st.session_state.performance_metrics['indexing_time'] = indexing_time
                                     else: st.error("Index creation failed."); st.stop()
                                 else: st.error("Node creation failed."); st.stop()
                              else: st.error("Cannot index: No parsed elements."); st.stop()
                         else: 
                             index_end_time = time.time()
                             index_duration = index_end_time - index_start_time
                             st.success("FAISS index loaded from cache.")
                             st.session_state.indexing_done = True
                             # Store performance metrics
                             st.session_state.performance_metrics['indexing_time'] = index_duration

                # QUERY ENGINE SETUP
                if st.session_state.indexing_done and st.session_state.query_engine is None:
                     with query_engine_status_placeholder.container(), st.spinner("Setting up Query Engine..."):
                         qe_start_time = time.time()
                         st.session_state.query_engine = setup_query_engine(st.session_state.faiss_index, VOYAGE_API_KEY, top_k=similarity_top_k, rerank_top_n=rerank_top_n)
                         qe_duration = time.time() - qe_start_time
                         if st.session_state.query_engine: 
                             st.success("Query Engine ready.")
                             # Store performance metrics
                             st.session_state.performance_metrics['query_engine_setup_time'] = qe_duration
                         else: st.error("Query Engine setup failed."); st.stop()
                
                # Calculate and store total processing time
                overall_duration = time.time() - overall_start_time
                st.session_state.performance_metrics['total_processing_time'] = overall_duration
                
                st.rerun()

        # --- Display Final Status ---
        # Shows the current status of the RAG pipeline stages.
        if st.session_state.parsing_done: parsing_status_placeholder.success("✅ Parsed")
        else: parsing_status_placeholder.warning("⏳ Parsing Pending")
        if st.session_state.indexing_done: indexing_status_placeholder.success("✅ Indexed")
        else: indexing_status_placeholder.warning("⏳ Indexing Pending")
        if st.session_state.query_engine: query_engine_status_placeholder.success("✅ Query Engine Ready")
        else: query_engine_status_placeholder.warning("⏳ Query Engine Pending")

        # --- Display Performance Metrics ---
        if st.session_state.query_engine and st.session_state.performance_metrics:
            with metrics_placeholder.container():
                st.caption("Pipeline Performance Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Elements", st.session_state.performance_metrics.get('elements_count', 0))
                    st.metric("Parse Time", f"{st.session_state.performance_metrics.get('parsing_time', 0):.2f}s")
                with metrics_col2:
                    st.metric("Chunks", st.session_state.performance_metrics.get('nodes_count', 0))
                    st.metric("Indexing Time", f"{st.session_state.performance_metrics.get('indexing_time', 0):.2f}s")
                with metrics_col3:
                    st.metric("Total Time", f"{st.session_state.performance_metrics.get('total_processing_time', 0):.2f}s")
        
        # --- Query Interface with Fixed Bottom Input ---
        if st.session_state.query_engine:
            process_button_placeholder.empty()  # Hide processing button
            
            # Create a three-part layout:
            # 1. Top part for document status and metrics
            # 2. Middle part for chat history (scrollable)
            # 3. Bottom part fixed for input

            st.divider()
            
            # Container for chat history (will be filled before the input section)
            chat_history_container = st.container()
            
            # Fixed divider
            st.divider()
            
            # Fixed input container at the bottom
            input_container = st.container()
            with input_container:
                st.subheader("Ask a Question")
                query_col1, query_col2 = st.columns([4, 1])
                with query_col1:
                    user_query = st.text_area("Your query:", height=100, key="query_area")
                with query_col2:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    submit_query = st.button("Get Answer", key="get_answer_button", use_container_width=True)
                
                # Add formatting hints
                with st.expander("Formatting Options", expanded=False):
                    st.markdown("""
                    Include these keywords in your query for special formatting:
                    - Add `format json` for JSON response
                    - Add `format matrix` or `format matrice` for tabular response
                    - Add `format list` or `format liste` for a numbered list
                    """)
            
            # Handle query submission
            if submit_query and user_query:
                # Add the query to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Reconfigure engine if sliders changed
                if (st.session_state.query_engine._retriever.similarity_top_k != similarity_top_k or
                    st.session_state.query_engine._node_postprocessors[0].top_n != rerank_top_n):
                     with st.spinner("Reconfiguring query engine..."):
                         logger.info(f"Parameters changed. Reconfiguring engine with K={similarity_top_k}, N={rerank_top_n}")
                         st.session_state.query_engine = setup_query_engine(st.session_state.faiss_index, VOYAGE_API_KEY, top_k=similarity_top_k, rerank_top_n=rerank_top_n)
                     if not st.session_state.query_engine: st.error("Failed to update query engine."); st.stop()
                
                # Process query with status indicator
                query_status = st.status(f"Processing: '{user_query[:60]}...'", expanded=False)
                source_expander_container = st.container() 
                
                try:
                    with query_status:
                        st.write("1. Retrieving relevant chunks...")
                        query_start_time = time.time()
                        
                        # Get chat history for context
                        chat_context = ""
                        if len(st.session_state.chat_history) > 2:
                            recent_history = st.session_state.chat_history[:-1]
                            if recent_history:
                                chat_context = "\nChat history context:\n"
                                for msg in recent_history[-4:]:
                                    chat_context += f"{msg['role'].capitalize()}: {msg['content']}\n"
                        
                        # Prepare query with chat history context and format instructions
                        base_query = f"{chat_context}\n\nCurrent query: {user_query}" if chat_context else user_query
                        enhanced_query = enhance_query_with_formatting(base_query)
                        
                        # Process the query
                        st.write("2. Generating response via Query Engine...")
                        # Create QueryBundle manually to ensure correct query structure
                        query_bundle = QueryBundle(query_str=enhanced_query)
                        response_obj = st.session_state.query_engine.query(query_bundle)
                        
                        if response_obj:
                            final_response_str = response_obj.response
                            source_nodes_for_display = response_obj.source_nodes
                            st.write("3. Response generated.")
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": final_response_str})
                        else: 
                            final_response_str = "Error: No response from engine."
                            source_nodes_for_display = []

                        query_end_time = time.time(); query_time = query_end_time - query_start_time
                    
                    # Update query status
                    query_status.update(label=f"Processing complete! ({query_time:.2f}s)", state="complete", expanded=False)

                    # Sources expander (outside status)
                    with source_expander_container:
                        if source_nodes_for_display:
                            with st.expander("Show Sources Used", expanded=False):
                                source_nodes_for_display.sort(key=lambda n: n.score if n.score is not None else 0.0, reverse=True)
                                st.write(f"{len(source_nodes_for_display)} source(s) after reranking:")
                                for i, node in enumerate(source_nodes_for_display):
                                    page = node.metadata.get('page_number', '?')
                                    elem_type = node.metadata.get('original_element_type', 'Unknown')
                                    score = node.score if node.score is not None else 0.0
                                    col1, col2 = st.columns([1, 4])
                                    with col1: st.info(f"**Source {i+1}**\nPg: {page} ({elem_type})\nScore: {score:.4f}")
                                    with col2: st.caption(f"Content Snippet:\n```text\n{node.text[:350].strip()}...\n```", unsafe_allow_html=False)
                
                except Exception as query_e:
                    query_end_time = time.time(); query_time = query_end_time - query_start_time
                    st.error(f"An error occurred during query: {type(query_e).__name__}")
                    logger.error("Query failed", exc_info=True)
                    query_status.update(label=f"Error after {query_time:.1f}s: {type(query_e).__name__}", state="error", expanded=True)
                    st.code(traceback.format_exc())
                
                # Force refresh to update chat history display
                st.rerun()
            
            # Fill the chat history container (after processing any new message)
            with chat_history_container:
                if st.session_state.chat_history:
                    st.subheader("Conversation")
                    
                    # Display each message in the chat history
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            # User message with blue background
                            user_container = st.container(border=True)
                            with user_container:
                                col1, col2 = st.columns([1, 20])
                                with col1:
                                    st.markdown("👤")
                                with col2:
                                    st.info(message['content'])
                        else:
                            # Assistant message with green background
                            assistant_container = st.container(border=True)
                            with assistant_container:
                                col1, col2 = st.columns([1, 20])
                                with col1:
                                    st.markdown("🤖")
                                with col2:
                                    st.success(message['content'])
                else:
                    st.info("No conversation yet. Ask a question to start!")

    else:  # Welcome screen
         col1, col2 = st.columns([1,2])
         with col1: st.image("https://img.freepik.com/free-vector/file-searching-concept-illustration_114360-4690.jpg", width=300)
         with col2:
             st.subheader("Welcome to the Temelion RAG Demo!")
             st.write("""
             Upload a technical PDF document in the sidebar to get started.
             - The system will parse and index the document content.
             - Ask questions about the document.
             - Adjust retrieval parameters in the sidebar.
             """)
             st.caption("Click 'Process Document' after uploading.")
             
         # Add note about PyTorch errors
         st.info("""
         Note: If you see PyTorch 'path.path' errors in the console logs, these are harmless warnings from Streamlit's hot-reload system.
         To eliminate these warnings, run Streamlit with: `STREAMLIT_RUN_ON_SAVE=false streamlit run temelion_rag_fixed.py`
         """)

# === Entry Point ===
if __name__ == "__main__":
    # Centralised API key check at start
    if not VOYAGE_API_KEY or not ANTHROPIC_API_KEY:
        st.error("FATAL ERROR: API keys missing. Please configure VOYAGE_API_KEY and ANTHROPIC_API_KEY.")
        logger.critical("API Keys missing. Application cannot start.")
        st.stop()
    elif not FAISS_AVAILABLE:
         st.error("FATAL ERROR: FAISS library not found. Please install `faiss-cpu` or `faiss-gpu`.")
         logger.critical("FAISS library missing. Application cannot start.")
         st.stop()
    else:
         run_streamlit_app()