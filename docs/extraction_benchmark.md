# Extraction Benchmark: PyMuPDF vs Docling

Benchmark performed on: `data/papers/1706.03762v7.pdf` (*Attention Is All You Need*, 15 pages).

---

## 1. Quantitative Comparison

| Metric | PyMuPDF (`fitz`) | Docling |
| :--- | :--- | :--- |
| **Execution Time** | **0.08 s** | **22.44 s** |
| **Total Characters** | 39,512 | 48,968 (+23.9% richer content) |
| **Total Lines** | 1,062 (fragmented wrapped text) | 344 (structured markdown paragraphs) |
| **Headings Detection** | Plain text string with line breaks | Semantic Markdown headings (`## Attention Is All You Need`) |
| **Table Formatting** | Broken text lines, misaligned columns | **Clean Markdown tables (`| Col | Col |`)** |
| **Layout Awareness** | Top-to-bottom raw stream | Multi-column aware, flow-preserving |

---

## 2. Qualitative Sample Comparison

### A. Title & Author Block
- **PyMuPDF**:
  ```text
  Attention Is All You Need
  Ashish Vaswani∗
  Google Brain
  avaswani@google.com
  Noam Shazeer∗
  Google Brain
  ```
- **Docling**:
  ```markdown
  ## Attention Is All You Need

  Ashish Vaswani∗ Google Brain avaswani@google.com Noam Shazeer∗ Google Brain noam@google.com
  ```

### B. Table Representation (Table 1: Complexity Comparison)
- **PyMuPDF** (unstructured, column headers mixed with cells):
  ```text
  Layer Type
  Complexity per Layer
  Sequential
  Maximum Path Length
  Operations
  Self-Attention
  O(n2 · d)
  O(1)
  O(1)
  ```
- **Docling** (native structured table):
  ```markdown
  | Layer Type     | Complexity per Layer | Sequential Operations | Maximum Path Length |
  |----------------|----------------------|-----------------------|---------------------|
  | Self-Attention | O(n2 · d)            | O(1)                  | O(1)                |
  | Recurrent      | O(n · d2)            | O(n)                  | O(n)                |
  ```

---

## 3. Engineering Assessment & Recommendation

- **Docling** is ~280× slower (22s vs 0.08s), but text extraction for the 3 papers is an **offline / one-time preprocessing step**, not an online user-facing pipeline step.
- The structured Markdown output from Docling makes a **huge difference** for RAG retrieval and chunking:
  1. Tables stay intact as tables instead of scrambled strings.
  2. Multi-column academic layouts don't bleed across columns.
  3. Chunking algorithms can split on markdown headings (`##`, `###`) cleanly.

