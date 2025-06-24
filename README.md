# PDF RAG Chatbot using Llama Index, VoyageAI, and Anthropic

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interactively ask questions about their content. The system leverages Llama Index for the RAG pipeline, VoyageAI for high-performance embeddings and reranking, Anthropic's Claude model for generation, FAISS for efficient vector storage, and Streamlit for the user interface.


<img width="1277" alt="Technical-Documents-RAG-1" src="https://github.com/user-attachments/assets/9aa52a8d-c9ac-4b8b-9bc9-89a1bdbdb981" />


<img width="1277" alt="Technical-Documents-RAG-2" src="https://github.com/user-attachments/assets/fe92da21-6c63-4756-b9c4-55425a98f4ec" />




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
*   **Modular Codebase:** Application logic is split into dedicated modules for better organisation and maintainability. 

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
    *   PyMuPDF (`pymupdf`): *Core PDF text/image extraction*. 
    *   Camelot (`camelot-py[cv]`) - *Table extraction*
    *   Pytesseract - *Python wrapper for Tesseract-OCR Engine. * - *Requires Tesseract OCR engine*
    *   Pillow (PIL) - *Image handling.*
    *   **Poppler:** - *PDF rendering utilities (required by `pdf2image` used indirectly for OCR page conversion).*
    *   **Tesseract OCR Engine:** - *External tool for Optical Character Recognition on images.*
    *   **Ghostscript:** - *External tool often required by Camelot for PDF processing.*
    *   PyMuPDF (`pymupdf`): *Core PDF text/image extraction*. 
    *   Camelot (`camelot-py[cv]`) - *Table extraction*
    *   Pytesseract - *Python wrapper for Tesseract-OCR Engine. * - *Requires Tesseract OCR engine*
    *   Pillow (PIL) - *Image handling.*
    *   **Poppler:** - *PDF rendering utilities (required by `pdf2image` used indirectly for OCR page conversion).*
    *   **Tesseract OCR Engine:** - *External tool for Optical Character Recognition on images.*
    *   **Ghostscript:** - *External tool often required by Camelot for PDF processing.*
*   **Data Handling:** Pandas
*   **GPU Check:** PyTorch (`torch`)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Miniconda or Anaconda:** Required for managing the Python environment and dependencies.
    *   [Miniconda Installation Guide](https://docs.conda.io/projects/miniconda/en/latest/)

2.  **Git:** For cloning the repository.
    *   [Git Website](https://git-scm.com/)

3.  **Poppler:** Required by libraries like `pdf2image` to interact with PDF files (e.g., convert pages to images for OCR).
    *   **Windows:**
        *   Download the latest Windows binaries from [Poppler for Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases/). (e.g., `Release-24.07.0-0`).
        *   Extract the archive to a permanent location (e.g., `C:\Program Files\poppler-24.07.0`).
        *   **Important:** Add the `bin\` subdirectory from the extracted archive (e.g., `C:\Program Files\poppler-24.07.0\Library\bin`) to your system's **PATH environment variable**. Restart your terminal/IDE after changing the PATH.
    *   **macOS (using Homebrew):**
        ```bash
        brew install poppler
        ```
        *(Homebrew usually handles the PATH configuration.)*
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update && sudo apt install poppler-utils
        ```
        *(The package manager usually handles the PATH configuration.)*
    *   **Linux (Fedora/CentOS):**
        ```bash
        sudo dnf install poppler-utils
        ```
        *(The package manager usually handles the PATH configuration.)*

4.  **Tesseract OCR Engine:** Essential for extracting text from images within PDFs.
    *   **Windows:**
        *   Download an installer, e.g., from the [Tesseract at UB Mannheim project](https://github.com/UB-Mannheim/tesseract/wiki) or using `winget install --id UB-Mannheim.TesseractOCR`.
        *   During installation, ensure you select the necessary language packs (at least **English** and **French** for this project: `eng`, `fra`).
        *   **Important:** Add the Tesseract installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system's **PATH environment variable**. Restart your terminal/IDE after changing the PATH.
    *   **macOS (using Homebrew):**
        ```bash
        brew install tesseract tesseract-lang
        ```
        *(This installs the engine and common language packs. Homebrew usually handles the PATH.)*
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra
        ```
        *(Installs the engine and specific language packs. PATH is usually handled.)*
    *   **Linux (Fedora/CentOS):**
        ```bash
        sudo dnf install tesseract tesseract-langpack-eng tesseract-langpack-fra
        ```
        *(Installs the engine and specific language packs. PATH is usually handled.)*
    *   *Verify Installation:* You can try running `tesseract --version` and `tesseract --list-langs` in your terminal to check the installation and available languages.

5.  **Ghostscript (Highly Recommended):** Camelot often relies on Ghostscript for table extraction, especially from certain PDF types. Install it to avoid potential errors.
    *   [Ghostscript Downloads](https://www.ghostscript.com/releases/gsdnld.html)
    *   Alternatively, use your package manager (e.g., `brew install ghostscript`, `sudo apt install ghostscript`).


## Installation

Follow these steps to set up the project locally:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yvesngouateu/Technical-Documents-RAG.git # Or your repository URL
    cd Technical-Documents-RAG
    ```

2.  **Create and Activate the Conda Environment:**
    This command uses the provided `environment.yml` file to install all necessary Python packages with the correct versions.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Configure API Keys (CRITICAL & SECURELY):**
    This application requires API keys for Voyage AI and Anthropic. **Do NOT hardcode these keys.** The script (`config.py`) reads them from environment variables. You must set these variables in your system: 

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
    conda activate tech_doc_rag # Check the 'name:' field in environment.yml for the exact environment name
    ```

2.  **Run the Streamlit Application:** 
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

*   **API Keys:** Must be configured via environment variables (`VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`) and are loaded in `config.py`. 
*   **Models & Parameters:** Settings like model names, embedding dimensions, parsing thresholds, etc., are centralised in `config.py`. 
*   **Retrieval Parameters:** `K` (similarity_top_k) and `N` (rerank_top_n) can be adjusted dynamically via sliders in the Streamlit UI sidebar (defined in `ui.py`). 
*   **File Paths:** Directories for uploads (`uploaded_pdfs/`), parsing cache (`cache/`), and FAISS index cache (`faiss_index_cache/`) are defined in `config.py` and created automatically if they don't exist. These directories are gitignored.

## Project Structure 

The project follows a modular structure:

```text
Technical-Documents-RAG/
├── assets/                  # UI images (logos, icons)
│   ├── tech-icon.png.png
│   ├── ai_building_hand.png
│   └── ai_building_large.png
├── cache/                   # (Gitignored) Cached parsed PDF data (*.pkl)
├── faiss_index_cache/       # (Gitignored) Cached FAISS index files
├── uploaded_pdfs/           # (Gitignored) Temporary storage for uploaded PDFs
│
├── config.py                # Central configuration (API keys, models, paths, thresholds)
├── pdf_parser.py            # PDF parsing logic (PyMuPDF, Camelot, Tesseract)
├── cache_utils.py           # Functions for saving/loading parsed data cache
├── indexing.py              # Node creation, embedding, FAISS index build/load logic
├── query_engine.py          # Setup for the LlamaIndex RetrieverQueryEngine
├── ui.py                    # Streamlit UI components and layout functions
├── main.py                  # Main application entry point, orchestrates modules and state
│
├── .gitignore                             # Specifies intentionally untracked files by Git
├── environment.yml                   # Conda environment definition with dependencies
├── LICENSE                                   # Project licence file (Consider adding one!) (Consider adding one!)
├── README.md                               # This documentation file
└── .env                                         # (Gitignored - if using this method) API keys
``````

To improve performance, the application implements caching:

*   **Parsed Elements:** When a PDF is successfully parsed, its extracted elements (text blocks, tables, OCR results) are saved as a `.pkl` file in the `cache/` directory. If the same PDF is uploaded again, these cached elements are loaded instead of re-parsing.
*   **FAISS Index:** After generating embeddings and building the FAISS index for a document, the index is persisted to a subdirectory within `faiss_index_cache/`. If the same document is processed again, the pre-built index is loaded, skipping the potentially time-consuming embedding generation step.

The **"Force Re-Parse & Re-Index"** button in the sidebar allows you to manually bypass and delete the cache and index for the currently loaded document.

## Troubleshooting

*   **TesseractNotFoundError:** Ensure Tesseract OCR is correctly installed AND its installation directory (containing the `tesseract` executable) is included in your system's PATH environment variable.
*   **Ghostscript Errors / Camelot Failures:** Table extraction might fail if Ghostscript is missing or not found. Ensure it is installed and accessible. Check Camelot documentation for specific dependency issues.
*   **API Key Errors:** Double-check that you have correctly set the `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` environment variables and that they are accessible to the running script (`main.py` which imports `config.py`). Restart your terminal or IDE after setting persistent variables. <!-- MODIFIED -->
*   **FAISS Errors:** Ensure you have the correct FAISS package installed (`faiss-cpu` or `faiss-gpu`) as listed in `environment.yml`. GPU usage requires compatible Nvidia drivers and CUDA toolkit installed, although the script falls back to CPU if GPU fails.
*   **Import Errors:** If you get errors like `ModuleNotFoundError`, ensure your Conda environment (`temelion-rag-env`) is activated and that all dependencies from `environment.yml` were installed correctly. Also, check that you are running `streamlit run main.py` from the root directory (`Technical-Documents-RAG`). <!-- NEW SECTION -->

## Licence

[MIT Licence]
