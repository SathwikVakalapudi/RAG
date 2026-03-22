import pdfplumber

def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    page_texts.append({"text": text, "page": page_num})
            return page_texts, None if page_texts else ([], "No text extracted")
    except Exception as e:
        return [], f"Error reading PDF: {e}"
