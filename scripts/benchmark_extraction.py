"""Benchmark comparing PyMuPDF vs Docling extraction on the research papers."""

import sys
import time
from pathlib import Path

# Target paper for benchmark (Attention Is All You Need)
PDF_PATH = Path("/Users/adity/Repos/vaani/data/papers/1706.03762v7.pdf")
OUTPUT_DIR = Path("/Users/adity/Repos/vaani/data/benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_pymupdf():
    import fitz  # PyMuPDF
    print("\n--- Running PyMuPDF (fitz) ---")
    start_time = time.perf_counter()
    doc = fitz.open(str(PDF_PATH))
    num_pages = len(doc)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    full_text = "\n".join(text_parts)
    elapsed = time.perf_counter() - start_time

    out_file = OUTPUT_DIR / "pymupdf_output.txt"
    out_file.write_text(full_text, encoding="utf-8")
    
    print(f"PyMuPDF completed in: {elapsed:.2f}s")
    print(f"Total pages: {num_pages}")
    print(f"Character count: {len(full_text)}")
    print(f"Line count: {len(full_text.splitlines())}")
    return elapsed, len(full_text), full_text

def benchmark_docling():
    from docling.document_converter import DocumentConverter
    print("\n--- Running Docling ---")
    start_time = time.perf_counter()
    converter = DocumentConverter()
    result = converter.convert(str(PDF_PATH))
    markdown_text = result.document.export_to_markdown()
    elapsed = time.perf_counter() - start_time

    out_file = OUTPUT_DIR / "docling_output.md"
    out_file.write_text(markdown_text, encoding="utf-8")

    print(f"Docling completed in: {elapsed:.2f}s")
    print(f"Character count: {len(markdown_text)}")
    print(f"Line count: {len(markdown_text.splitlines())}")
    return elapsed, len(markdown_text), markdown_text

if __name__ == "__main__":
    p_time, p_chars, p_text = benchmark_pymupdf()
    try:
        d_time, d_chars, d_text = benchmark_docling()
    except Exception as e:
        print(f"Docling error: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("EXTRACTION COMPARISON (Attention Is All You Need - Page 1 Sample)")
    print("="*50)
    print("\n[PyMuPDF First 500 Chars]:\n")
    print(p_text[:500])
    print("\n" + "-"*40)
    print("\n[Docling First 500 Chars]:\n")
    print(d_text[:500])
    print("="*50)

