"""
PDF Parsing Module for Technical Document RAG Demo
========================================

This module provides functionality for parsing PDF documents using a hybrid approach:
- PyMuPDF for text extraction and image identification
- Pytesseract for OCR on identified images
- Camelot for table extraction with heuristic detection

The main class, PDFParser, processes PDFs and extracts structured content
with detailed metadata for further processing and indexing.


"""

import os
import logging
import re
import fitz  # PyMuPDF
import pytesseract  # For OCR
from PIL import Image
import io  # For image conversion
import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st

# Import configuration
from config import (
    OCR_THRESHOLD_CHARS, MIN_TABLE_ROWS, MIN_TABLE_COLS, 
    TABLE_KEYWORDS, TABLE_HEURISTIC_MIN_LINES, 
    TABLE_HEURISTIC_NUMERIC_RATIO, CAMELOT_AVAILABLE
)

# Get module-level logger
logger = logging.getLogger(__name__)

class PDFParser:
    """
    A class to parse PDF documents using a hybrid approach:
    - PyMuPDF for text blocks and image identification.
    - Pytesseract for OCR on identified images.
    - Heuristics and Camelot for controlled table extraction.
    """
    def __init__(self):
        """
        Initialise the PDFParser.
        Logging and warnings are configured globally in the config module.
        """
        pass

    def _extract_text_from_image_ocr(self, img_bytes: bytes) -> str:
        """
        Extracts text from an image using OCR (Optical Character Recognition).
        
        Args:
            img_bytes (bytes): The image data as bytes.
            
        Returns:
            str: The extracted text or an empty string if OCR fails.
        """
        #Using io.BytesIO to handle image bytes and perform OCR
        #and PIL to open the image
        try:
            image = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(image, lang='fra+eng')
            return text.strip()
        except Exception as e:
            #Log with Streamlit in mind, avoid excessive console noise
            st.sidebar.warning(f"OCR failed for an image: {e}", icon="⚠️")
            return ""

    def _is_likely_table_heuristic(self, text_block: str) -> bool:
        """
        Uses heuristics to determine if a text block likely contains a table.
        
        Args:
            text_block (str): The text block to analyse.
            
        Returns:
            bool: True if the text block likely contains a table, False otherwise.
        """
        #Check for keywords indicating a table
        #and for a sufficient number of lines and numeric/indented content
        text_lower = text_block.lower()
        if any(keyword in text_lower for keyword in TABLE_KEYWORDS): 
            return True
        #Check for a minimum number of lines
        #and a sufficient ratio of numeric or indented lines   
        lines = text_block.strip().split('\n')
        if len(lines) > MIN_TABLE_ROWS:
            numerical_or_indented_lines = 0
            for line in lines:  
                #Check if the line is indented or starts with a digit
                #or a bullet point
                stripped_line = line.strip()
                if stripped_line and (stripped_line[0].isdigit() or 
                                     stripped_line[0] in ['-','*','•'] or 
                                     line.startswith("   ")):
                    numerical_or_indented_lines += 1
            if numerical_or_indented_lines > len(lines) / 2: 
                return True
                
        return False

    def _extract_specific_tables_camelot(self, file_path: str, page_number: int) -> List[str]:
        """
        Extracts tables from a specific page of a PDF using Camelot.
        
        Args:
            file_path (str): Path to the PDF file.
            page_number (int): Page number to extract tables from.
            
        Returns:
            List[str]: List of extracted tables in Markdown format.
        """
        validated_tables_markdown: List[str] = []
        
        # Return empty list if Camelot is not available
        if not CAMELOT_AVAILABLE:
            return validated_tables_markdown
        # Attempt to extract tables using Camelot
        # Use 'lattice' flavor first, then 'stream' if no tables found
        # Suppress stdout to avoid cluttering Streamlit output
        # and handle exceptions gracefully
        # and log less verbosely for Streamlit run  
        try:
            import camelot
            tables = camelot.read_pdf(file_path, pages=str(page_number), flavor='lattice', suppress_stdout=True)
            if tables.n == 0:
                tables = camelot.read_pdf(file_path, pages=str(page_number), flavor='stream', suppress_stdout=True)
                
            # Process tables (same validation logic)
            # and convert to Markdown format
            # and log less verbosely for Streamlit run
            for i, table in enumerate(tables):
                table_df = table.df
                if isinstance(table_df, pd.DataFrame) and not table_df.empty and \
                   table_df.shape[0] >= MIN_TABLE_ROWS and table_df.shape[1] >= MIN_TABLE_COLS:
                    # Check for numeric content ratio
                    try:
                        markdown_table = table_df.to_markdown(index=False)
                        validated_tables_markdown.append(
                            f"\n--- TABLE Camelot_{i+1} (Page {page_number}) ---\n{markdown_table}\n--- END TABLE Camelot_{i+1} ---\n"
                        )
                    except Exception as md_e: 
                         # Ignore markdown conversion errors silently for Streamlit
                        pass
        except Exception as e:
            # Log less verbosely for Streamlit run
            # and handle specific errors
             if "ghostscript" in str(e).lower():
                 logging.error("Ghostscript error during Camelot extraction.")
                 st.sidebar.error("Ghostscript n'est pas trouvé. L'extraction de tables Camelot peut échouer.", icon="❌")
             # else: logging.warning(f"Camelot error p{page_number}: {e}")
        return validated_tables_markdown

    def parse_pdf(self, file_path: str, progress_bar: Optional[st.progress] = None) -> List[Dict[str, Any]]:
        """
        Main method to parse a PDF document with a hybrid approach.
        
        Args:
            file_path (str): Path to the PDF file.
            progress_bar (Optional[st.progress]): Streamlit progress bar for visual feedback.
            
        Returns:
            List[Dict[str, Any]]: List of extracted elements with their metadata.
        """
        # ---  Initialize Variables ---
        # List to store extracted elements
        # Each element is a dictionary with text content and metadata
        # Metadata includes source file, page number, block type, and bounding box
        # and other relevant information
        extracted_elements: List[Dict[str, Any]] = []
        doc = None
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            st.error(f"Erreur à l'ouverture du PDF {file_path}: {e}")
            return []

        total_pages = len(doc)
        st.info(f"Démarrage du parsing : {os.path.basename(file_path)} ({total_pages} pages)")

        # --- Parse each page ---
        # Iterate through each page of the PDF
        # and extract text blocks, images, and perform OCR
        # and extract tables using Camelot if heuristics indicate a table
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_number_actual = page_num + 1
            table_extraction_attempted_on_page = False

            # Update progress bar if provided
            if progress_bar:
                 progress_value = (page_num + 1) / total_pages
                 progress_bar.progress(progress_value, text=f"Parsing Page {page_number_actual}/{total_pages}")

            # --- 1. Extract Text Blocks ---
            # Extract text blocks using PyMuPDF
            # and check for table heuristics
            try:
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
                if blocks and 'blocks' in blocks:
                    for block in blocks.get('blocks', []):
                        # Text blocks are identified by type 0
                        # and contain text content
                        if block['type'] == 0: 
                            # Extract block_text and bbox
                            # and check for empty text
                            # and handle empty blocks
                            block_text = ""
                            for line in block.get('lines', []):
                                for span in line.get('spans', []): 
                                    block_text += span['text'] + " "
                                block_text += "\n"
                            block_text = block_text.strip()
                            # Check if block_text is not empty
                        
                            if block_text:
                                # Extract bounding box coordinates
                                # and round them to 2 decimal places
                                # and create metadata dictionary
                                bbox = block['bbox']
                                metadata = {
                                    "source_file": os.path.basename(file_path), 
                                    "page_number": page_number_actual,
                                    "block_type": "text", 
                                    "bbox": [round(c, 2) for c in bbox] 
                                }
                                # Append text block to extracted_elements
                                extracted_elements.append({
                                    "page_number": page_number_actual, 
                                    "element_type": "TextBlock",
                                    "text_content": block_text, 
                                    "metadata": metadata 
                                })

                                # --- 2. Heuristic Table Detection ---
                                # Check if the block is likely a table
                                # and if so, attempt to extract tables using Camelot
                                if not table_extraction_attempted_on_page and self._is_likely_table_heuristic(block_text):
                                    validated_tables_md = self._extract_specific_tables_camelot(file_path, page_number_actual)
                                    table_extraction_attempted_on_page = True
                                    #process each validated table
                                    # and append to extracted_elements
                                    for t_idx, table_md in enumerate(validated_tables_md):
                                        table_metadata = {
                                            "source_file": os.path.basename(file_path), 
                                            "page_number": page_number_actual,
                                            "block_type": "table", 
                                            "extraction_method": "Camelot" 
                                        }
                                        extracted_elements.append({
                                            "page_number": page_number_actual, 
                                            "element_type": "Table",
                                            "text_content": table_md, 
                                            "metadata": table_metadata 
                                        })
            except Exception as block_e:
                logging.error(f"Page {page_number_actual}: Error processing text blocks: {block_e}")

            # --- 3. Extract Images and Perform OCR ---
            try:
                # Extract images from the page
                # and perform OCR on each image
                # and append the OCR text to extracted_elements
                # and create metadata dictionary
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                     xref = img_info[0]
                     bbox = page.get_image_bbox(img_info).irect
                     base_image = doc.extract_image(xref)
                     image_bytes = base_image["image"]
                     ocr_text = self._extract_text_from_image_ocr(image_bytes)
                     if ocr_text:
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
            except Exception as img_e: 
                logging.error(f"Page {page_number_actual}: Error processing images/OCR: {img_e}")
            
            # --- 4. Fallback Page-Level OCR ---
            # If no text blocks or images were found on the page
            # and the content length is below the threshold, apply fallback OCR
            current_page_content_len = sum(
                len(el.get('text_content', '')) 
                for el in extracted_elements 
                if el.get('page_number') == page_number_actual
            )
            # Only if no images were OCRed already
            # and the content length is below the threshold
            # and no tables were extracted
            # and no text blocks were found
            # and the page is not empty
            if current_page_content_len < OCR_THRESHOLD_CHARS and not image_list: 
                warning_msg = (f"Page {page_number_actual}: Very low content ({current_page_content_len} chars), "
                              f"no images found. Applying fallback full-page OCR.")
                logging.warning(warning_msg)
                try:
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    full_page_ocr_text = self._extract_text_from_image_ocr(img_bytes)
                    # Check if the OCR text is not empty
                    # and if so, append it to extracted_elements
                    # and create metadata dictionary
                    if full_page_ocr_text:
                        ocr_metadata = {
                            "source_file": os.path.basename(file_path), 
                            "page_number": page_number_actual, 
                            "block_type": "full_page_ocr"
                        }
                        extracted_elements.append({
                            "page_number": page_number_actual, 
                            "element_type": "FullPageOCR",
                            "text_content": full_page_ocr_text, 
                            "metadata": ocr_metadata
                        })
                except Exception as fallback_e: 
                    logging.error(f"Page {page_number_actual}: Fallback OCR failed: {fallback_e}")
            
            # --- Progress Update ---
            if page_number_actual % 10 == 0 or page_number_actual == total_pages: 
                print(f"  Processed page {page_number_actual}/{total_pages}")
            
        # --- 5. Cleanup and Final Sort ---
        # Close the document
        # and ensure the progress bar reaches 100%
        # and print completion message
        # and sort elements by page number and vertical position
        # and return the extracted elements
        if doc:
            doc.close()
        # Ensure progress bar reaches 100%
        if progress_bar: 
            progress_bar.progress(1.0, text="Parsing Terminé !")
        print("Parsing complete.")
        
        # Sort elements by page number and vertical position
        extracted_elements.sort(key=lambda x: (
            x.get('page_number', 0), 
            x.get('metadata', {}).get('bbox', [0, 0, 0, 0])[1]
        ))
        
        return extracted_elements
