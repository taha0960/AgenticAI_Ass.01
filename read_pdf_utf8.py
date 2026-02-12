import sys
import pypdf
import os

path = r'd:/TAHA_Agentic/hw1-phase-1-semantic-search-module-taha0960/HW1_Phase1_AgenticAI.pdf'
out_path = r'd:/TAHA_Agentic/hw1-phase-1-semantic-search-module-taha0960/pdf_content_utf8.txt'

try:
    reader = pypdf.PdfReader(path)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Number of pages: {len(reader.pages)}\n")
        for i, page in enumerate(reader.pages):
            f.write(f"--- Page {i+1} ---\n")
            f.write(page.extract_text() + "\n")
    print("Done")
except Exception as e:
    print(f"Error: {e}")
