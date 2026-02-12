import sys
import pypdf
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

path = r'd:/TAHA_Agentic/hw1-phase-1-semantic-search-module-taha0960/HW1_Phase1_AgenticAI.pdf'
print(f"Reading {path}")
if not os.path.exists(path):
    print("File not found!")
    sys.exit(1)

try:
    reader = pypdf.PdfReader(path)
    print(f"Number of pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        print(f"--- Page {i+1} ---")
        print(page.extract_text())
except Exception as e:
    print(f"Error: {e}")
