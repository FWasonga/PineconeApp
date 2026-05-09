# pdf_reader.py
# ------------------------------------
# Reads a PDF file page by page and
# returns all text as a single string
# ------------------------------------

from pypdf import PdfReader

def load_pdf(filepath: str) -> str:
    """
    Opens a PDF file and extracts all text from every page.
    
    Args:
        filepath: path to your PDF file e.g. "documents/myfile.pdf"
    
    Returns:
        A single string containing all text from the PDF
    """
    
    # Open the PDF — like opening a book
    reader = PdfReader(filepath)
    
    full_text = ""
    
    # Loop through every page and extract its text
    for page_number, page in enumerate(reader.pages):
        extracted = page.extract_text()
        
        # Some pages may return None (e.g. image-only pages)
        # We use 'or ""' to safely handle that
        full_text += (extracted or "") + "\n"
        
    print(f"✅ PDF loaded: {len(reader.pages)} pages, "
          f"{len(full_text)} characters extracted")
    
    return full_text