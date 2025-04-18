# PDF RAG Chatbot using Llama Index, VoyageAI, and Anthropic

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interactively ask questions about their content. The system leverages Llama Index for the RAG pipeline, VoyageAI for high-performance embeddings and reranking, Anthropic's Claude model for generation, FAISS for efficient vector storage, and Streamlit for the user interface.

![image](https://github.com/user-attachments/assets/96a1c0d2-2888-407b-9521-788c305487a0)



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
*   **Secure Configuration:** API keys are loaded from environment variables, avoiding hardcoding sensitive information.

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
    conda activate temelion # Check the 'name:' field in environment.yml for the exact environment name
    ```

3.  **Configure API Keys (CRITICAL & SECURELY):**
    This application requires API keys for Voyage AI and Anthropic. **Do NOT hardcode these keys in the script.** The script (`app.py`) reads them from environment variables. You must set these variables in your system:

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
    *   **Using a `.env` file (Alternative - Requires code modification):** If you prefer using a `.env` file, you would need to:
        1.  Install `python-dotenv` (`pip install python-dotenv` and add it to `environment.yml`).
        2.  Create a `.env` file in the project root (add it to `.gitignore`!).
        3.  Add `from dotenv import load_dotenv` and `load_dotenv()` near the start of `app.py`.
        *(Note: The current script does not implement this method out-of-the-box.)*

## Usage

1.  **Activate the Conda Environment:**
    ```bash
    conda activate temelion # Or your environment name
    ```

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    *   *(Optional)* To potentially suppress harmless PyTorch warnings related to Streamlit's hot-reloading, you can run:
        ```bash
        STREAMLIT_RUN_ON_SAVE=false streamlit run app.py
        ```

3.  **Interact with the Application:**
    *   The application will open in your web browser.
    *   Use the sidebar to **Upload a PDF document**.
    *   Click the **"Process Document (Parse & Index)"** button. Wait for the status indicators (Parsed, Indexed, Query Engine Ready) to show success (✅). This may take some time for the first processing of a document, especially embedding generation.
    *   Once the Query Engine is ready, use the **chat input area** at the bottom to ask questions about the document's content.
    *   View the generated answers and expand the **"Show Sources Used"** section to see the relevant text chunks retrieved from the document.
    *   Adjust the **"Nodes Retrieved (K)"** and **"Nodes Reranked (N)"** sliders in the sidebar to tune retrieval behaviour (changes apply on the next query).
    *   Use **"Force Re-Parse & Re-Index"** to clear caches and reprocess the current document from scratch.
    *   Use **"Clear Chat History"** to reset the conversation.
    *   Use **"Reset Entire System"** to clear everything and start fresh with a new document upload.

## Configuration

*   **API Keys:** Must be configured via environment variables (`VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`) as described in the Installation section. The application will fail to start if these are missing.
*   **Models:** Embedding, reranking, and LLM models are currently defined as constants within `app.py` (e.g., `VOYAGE_EMBEDDING_MODEL`, `ANTHROPIC_LLM_MODEL`).
*   **Retrieval Parameters:** `K` (similarity_top_k) and `N` (rerank_top_n) can be adjusted dynamically via sliders in the Streamlit UI sidebar.
*   **File Paths:** Directories for uploads (`uploaded_pdfs/`), parsing cache (`cache/`), and FAISS index cache (`faiss_index_cache/`) are defined in `app.py` and created automatically. These directories are gitignored.

## Project Structure
Junior_ML_Engineer/
├── assets/ # UI images (logos, icons)
│ ├── logo_temelion.png
│ ├── ai_building_hand.png
│ └── ai_building_large.png
├── cache/ # (Gitignored) Stores cached parsed PDF elements (*.pkl)
├── faiss_index_cache/ # (Gitignored) Stores persisted FAISS index files
├── uploaded_pdfs/ # (Gitignored) Temporary storage for uploaded PDFs
├── .gitignore # Specifies intentionally untracked files by Git
├── app.py # The main Streamlit application script
├── environment.yml # Conda environment definition with dependencies
├── LICENSE # Project licence file
├── README.md # This documentation file
└── .env # (Gitignored - if you choose this method) API keys


## Caching Behaviour

To improve performance, the application implements caching:

*   **Parsed Elements:** When a PDF is successfully parsed, its extracted elements (text blocks, tables, OCR results) are saved as a `.pkl` file in the `cache/` directory. If the same PDF is uploaded again, these cached elements are loaded instead of re-parsing.
*   **FAISS Index:** After generating embeddings and building the FAISS index for a document, the index is persisted to a subdirectory within `faiss_index_cache/`. If the same document is processed again, the pre-built index is loaded, skipping the potentially time-consuming embedding generation step.

The **"Force Re-Parse & Re-Index"** button in the sidebar allows you to manually bypass and delete the cache and index for the currently loaded document.

## Troubleshooting

*   **TesseractNotFoundError:** Ensure Tesseract OCR is correctly installed AND its installation directory (containing the `tesseract` executable) is included in your system's PATH environment variable.
*   **Ghostscript Errors / Camelot Failures:** Table extraction might fail if Ghostscript is missing or not found. Ensure it is installed and accessible. Check Camelot documentation for specific dependency issues.
*   **API Key Errors:** Double-check that you have correctly set the `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` environment variables and that they are accessible to the running script. Restart your terminal or IDE after setting persistent variables.
*   **FAISS Errors:** Ensure you have the correct FAISS package installed (`faiss-cpu` or `faiss-gpu`) as listed in `environment.yml`. GPU usage requires compatible Nvidia drivers and CUDA toolkit installed, although the script falls back to CPU if GPU fails.

## Licence

[MIT Licence]
