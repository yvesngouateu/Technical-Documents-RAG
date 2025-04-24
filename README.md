# PDF RAG Chatbot using Llama Index, VoyageAI, and Anthropic

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interactively ask questions about their content. The system leverages Llama Index for the RAG pipeline, VoyageAI for high-performance embeddings and reranking, Anthropic's Claude model for generation, FAISS for efficient vector storage, and Streamlit for the user interface.

![image](https://github.com/user-attachments/assets/033a748e-f5d5-4ee2-ad44-9e8bec8a6dd4)


## Key Features

*   **Multi-Modal PDF Parsing:** Extracts information from PDFs including:
    *   Text blocks (via PyMuPDF)
    *   Structured tables (via Camelot-py)
    *   Text embedded within images (via Tesseract OCR & PyMuPDF)
*   **Advanced RAG Pipeline:** Built with Llama Index, featuring:
    *   High-quality text embeddings using VoyageAI (`voyage-3-large`).
    *   Efficient vector search using FAISS (with optional GPU acceleration check).
    *   Contextual reranking using VoyageAI (`rerank-2`) to improve relevance.
    *   Response generation using Anthropic's Claude (`claude-3-7-sonnet-20250219`).
*   **Interactive User Interface:** Developed with Streamlit, allowing:
    *   Easy PDF document uploading.
    *   Control over retrieval (`K`) and reranking (`N`) parameters.
    *   Display of processing status (parsing, indexing, query engine readiness).
    *   Conversation history tracking.
    *   Display of source nodes used for generating answers.
    *   Performance metrics (parsing time, indexing time, etc.).
*   **Efficient Caching:** Caches parsed document elements and FAISS indices locally to significantly speed up processing for previously uploaded documents.
*   **Secure Configuration:** API keys are loaded from environment variables (via `config.py`), avoiding hardcoding sensitive information.
*   **Modular Codebase:** Application logic is split into dedicated modules for better organisation and maintainability. <!-- MODIFIED -->

## Technologies Used

*   **Programming Language:** Python 3.x
*   **Environment Management:** Miniconda / Conda
*   **Web Framework / UI:** Streamlit
*   **RAG Framework:** Llama Index
    *   `llama-index-core`
    *   `llama-index-embeddings-voyageai`
    *   `llama-index-llms-anthropic`
    *   `llama-index-vector-stores-faiss`
    *   `llama-index-postprocessor-voyageai-rerank`
*   **AI Models & Services:**
    *   Voyage AI (Embeddings & Reranking)
    *   Anthropic (Claude LLM)
*   **Vector Store:** FAISS (`faiss-cpu` or `faiss-gpu`)
*   **PDF Parsing:**
    *   PyMuPDF (`pymupdf`)
    *   Camelot (`camelot-py[cv]`) - *Requires Ghostscript*
    *   Pytesseract - *Requires Tesseract OCR engine*
    *   Pillow (PIL)
*   **Data Handling:** Pandas
*   **GPU Check:** PyTorch (`torch`)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Miniconda or Anaconda:** Required for managing the Python environment and dependencies. [Miniconda Installation Guide](https://docs.conda.io/projects/miniconda/en/latest/)
2.  **Git:** For cloning the repository. [Git Website](https://git-scm.com/)
3.  **Tesseract OCR Engine:** Essential for extracting text from images within PDFs.
    *   Installation instructions vary by OS: [Tesseract Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   **Important:** Ensure Tesseract is added to your system's PATH environment variable so the script can find it.
    *   Install necessary language packs (e.g., `eng` for English, `fra` for French) during or after installation. The script uses `lang='fra+eng'`.
4.  **Ghostscript (Highly Recommended):** Camelot often relies on Ghostscript for table extraction, especially from certain PDF types. Install it to avoid potential errors. [Ghostscript Downloads](https://www.ghostscript.com/releases/gsdnld.html)

## Installation

Follow these steps to set up the project locally:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yvesngouateu/Junior_ML_Engineer.git # Or your repository URL
    cd Junior_ML_Engineer
    ```

2.  **Create and Activate the Conda Environment:**
    This command uses the provided `environment.yml` file to install all necessary Python packages with the correct versions.
    ```bash
    conda env create -f environment.yml
    conda activate temelion-rag-env # Check the 'name:' field in environment.yml for the exact environment name
    ```

3.  **Configure API Keys (CRITICAL & SECURELY):**
    This application requires API keys for Voyage AI and Anthropic. **Do NOT hardcode these keys.** The script (`config.py`) reads them from environment variables. You must set these variables in your system: <!-- MODIFIED -->

    *   `VOYAGE_API_KEY`: Your API key from Voyage AI.
    *   `ANTHROPIC_API_KEY`: Your API key from Anthropic.

    **Methods to set environment variables:**

    *   **Temporary (Current Terminal Session):**
        *   *Linux/macOS:*
            ```bash
            export VOYAGE_API_KEY="your_voyage_ai_key_here"
            export ANTHROPIC_API_KEY="your_anthropic_key_here"
            ```
        *   *Windows (Command Prompt):*
            ```bash
            set VOYAGE_API_KEY=your_voyage_ai_key_here
            set ANTHROPIC_API_KEY=your_anthropic_key_here
            ```
        *   *Windows (PowerShell):*
            ```powershell
            $env:VOYAGE_API_KEY="your_voyage_ai_key_here"
            $env:ANTHROPIC_API_KEY="your_anthropic_key_here"
            ```
    *   **Persistent (Recommended for local development):** Add the `export` (Linux/macOS) or `set` / `$env:` (Windows) commands to your shell's profile script (e.g., `.bashrc`, `.zshrc`, `.profile`, or via System Environment Variables settings on Windows).
    *   **Using a `.env` file (Alternative):** You could modify `config.py` to use `python-dotenv` (requires installing it and adding `load_dotenv()` in `config.py`). Create a `.env` file (add to `.gitignore`!) with the keys.

## Usage

1.  **Activate the Conda Environment:**
    ```bash
    conda activate temelion-rag-env # Or your environment name
    ```

2.  **Run the Streamlit Application:** <!-- MODIFIED -->
    The application entry point is now `main.py`.
    ```bash
    streamlit run main.py
    ```
    *   *(Optional)* To potentially suppress harmless PyTorch warnings related to Streamlit's hot-reloading, you can run:
        ```bash
        STREAMLIT_RUN_ON_SAVE=false streamlit run main.py
        ```

3.  **Interact with the Application:**
    *   The application will open in your web browser.
    *   Use the sidebar to **Upload a PDF document**.
    *   Click the **"Démarrez"** (Start) button. Wait for the status indicators (Parsed, Indexed, Query Engine Ready) to show success (✅). This may take some time for the first processing of a document, especially embedding generation.
    *   Once the Query Engine is ready, use the **chat input area** at the bottom to ask questions about the document's content.
    *   View the generated answers and expand the **"View Sources"** section to see the relevant text chunks retrieved from the document.
    *   Adjust the **"Nodes Retrieved (K)"** and **"Nodes Reranked (N)"** sliders in the sidebar to tune retrieval behaviour (changes apply on the next query).
    *   Use **"🔄 Nettoyer les caches"** (Clear Caches) to clear caches and reprocess the current document from scratch.
    *   Use **"🔄 Nettoyer la discussion"** (Clear Discussion) to reset the conversation.
    *   Use **"❌ Réinitialiser le système"** (Reset System) to clear everything and start fresh with a new document upload.

## Configuration

*   **API Keys:** Must be configured via environment variables (`VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`) and are loaded in `config.py`. <!-- MODIFIED -->
*   **Models & Parameters:** Settings like model names, embedding dimensions, parsing thresholds, etc., are centralised in `config.py`. <!-- MODIFIED -->
*   **Retrieval Parameters:** `K` (similarity_top_k) and `N` (rerank_top_n) can be adjusted dynamically via sliders in the Streamlit UI sidebar (defined in `ui.py`). <!-- MODIFIED -->
*   **File Paths:** Directories for uploads (`uploaded_pdfs/`), parsing cache (`cache/`), and FAISS index cache (`faiss_index_cache/`) are defined in `config.py` and created automatically if they don't exist. These directories are gitignored.

## Project Structure <!-- MODIFIED -->

The project follows a modular structure:

Junior_ML_Engineer/

├── assets/ # Image assets for the UI
│ ├── logo_temelion.png
│ ├── ai_building_hand.png
│ └── ai_building_large.png
├── cache/ # (Gitignored) Cached parsed PDF data (*.pkl)
├── faiss_index_cache/ # (Gitignored) Cached FAISS index files
├── uploaded_pdfs/ # (Gitignored) Temporary storage for uploaded PDFs
│
├── config.py # Central configuration (API keys, models, paths, thresholds)
├── pdf_parser.py # PDF parsing logic (PyMuPDF, Camelot, Tesseract)
├── cache_utils.py # Functions for saving/loading parsed data cache
├── indexing.py # Node creation, embedding, FAISS index build/load logic
├── query_engine.py # Setup for the LlamaIndex RetrieverQueryEngine
├── ui.py # Streamlit UI components and layout functions
├── main.py # Main application entry point, orchestrates modules and state
│
├── .gitignore # Specifies intentionally untracked files by Git
├── environment.yml # Conda environment definition with dependencies
├── LICENSE # Project licence file
├── README.md # This documentation file
└── .env # (Gitignored - if using this method) API keys


## Caching Behaviour

## Caching Behaviour

To improve performance, the application implements caching using functions in `cache_utils.py`: <!-- MODIFIED -->

*   **Parsed Elements:** When a PDF is successfully parsed (`pdf_parser.py`), its extracted elements are saved as a `.pkl` file in the `cache/` directory (path generated by `cache_utils.get_cache_path`). If the same PDF is uploaded again, these cached elements are loaded (`cache_utils.load_parsed_elements`) instead of re-parsing.
*   **FAISS Index:** After generating embeddings and building the FAISS index (`indexing.py`), the index is persisted to a subdirectory within `faiss_index_cache/`. If the same document is processed again, the pre-built index is loaded (`indexing.load_faiss_index`), skipping the potentially time-consuming embedding generation step.

The **"🔄 Nettoyer les caches"** button in the sidebar allows you to manually bypass and delete the cache and index for the currently loaded document.

## Troubleshooting

*   **TesseractNotFoundError:** Ensure Tesseract OCR is correctly installed AND its installation directory (containing the `tesseract` executable) is included in your system's PATH environment variable.
*   **Ghostscript Errors / Camelot Failures:** Table extraction might fail if Ghostscript is missing or not found. Ensure it is installed and accessible. Check Camelot documentation for specific dependency issues.
*   **API Key Errors:** Double-check that you have correctly set the `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` environment variables and that they are accessible to the running script (`main.py` which imports `config.py`). Restart your terminal or IDE after setting persistent variables. <!-- MODIFIED -->
*   **FAISS Errors:** Ensure you have the correct FAISS package installed (`faiss-cpu` or `faiss-gpu`) as listed in `environment.yml`. GPU usage requires compatible Nvidia drivers and CUDA toolkit installed, although the script falls back to CPU if GPU fails.
*   **Import Errors:** If you get errors like `ModuleNotFoundError`, ensure your Conda environment (`temelion-rag-env`) is activated and that all dependencies from `environment.yml` were installed correctly. Also, check that you are running `streamlit run main.py` from the root directory (`Junior_ML_Engineer/`). <!-- NEW SECTION -->

## Licence

[MIT Licence]
