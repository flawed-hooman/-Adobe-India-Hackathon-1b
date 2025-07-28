# Challenge 1b: Multi-Collection PDF Analysis

## Overview
This project implements a pipeline to extract structured content from PDF documents, build hierarchical representations of the documents, and generate semantically ranked sections relevant to a user's task such as travel planning.

The system processes multiple document collections, each containing a set of PDF guides and a user-provided configuration JSON (with user role/persona and query). The output is a JSON report capturing key document sections and detailed content subsections, ranked according to relevance to the user query.

## Approach

### 1) PDF Parsing and Structural Extraction:
PDFs are parsed page-by-page using PyMuPDF (fitz). Text spans are extracted and clustered into lines, then grouped into blocks based on spatial layout and font styling. Blocks are scored heuristically considering font size, bold/italic flags, and layout proximity. Using heuristics and semantic checks, blocks are classified as headers (with levels H1/H2/H3) or as paragraphs. The document content is represented as a tree of nodes, capturing document hierarchy.

### 2) Semantic Embedding and Ranking:
Extracted headers from all documents are embedded using a transformer model (sentence-transformers, specifically all-MiniLM-L6-v2). A FAISS vector index is built over these embeddings for efficient similarity search. Given the user persona and specific query (job to be done), an embedding is generated and queried against the corpus. Top-ranked document sections are selected based on similarity scores.

### 3) Detailed Subsection Analysis:
Candidate paragraphs and multi-paragraph subsections are extracted under headers. These are filtered by relevance to keywords in the user query. Subsections are embedded and reranked to select key content chunks. Relevant subsections are trimmed and prepared for final output.

### 4) Outputs:
Per-collection JSON reports summarize the important sections and relevant sub-contents. (Optional) JSON files represent the detailed document tree structures for each PDF.


## Project Structure
```
Challenge_1b/
├── input/       
│    ├── Collection 1/                    # Travel Planning
│    │   ├── PDFs/                       # South of France guides
│    │   └── challenge1b_input.json      # Input configuration
│    ├── Collection 2/                    # Adobe Acrobat Learning
│    │   ├── PDFs/                       # Acrobat tutorials
│    │   └── challenge1b_input.json      # Input configuration
│    └── Collection 3/                    # Recipe Collection
│        ├── PDFs/                       # Cooking guides
│        └── challenge1b_input.json      # Input configuration
├── output/
│    ├── output_Collection 1     # Analysis results
│    ├── output_Collection 2     # Analysis results
│    └── output_Collection 3     # Analysis results
├── .dockerignore
├── Dockerfile
├── main.py
├── requirements.txt       
└── README.md
```


## Models and Libraries Used
- **PyMuPDF (fitz)** : For fast PDF parsing, layout analysis, and text extraction.
- **sentence-transformers (all-MiniLM-L6-v2 model)**: To generate sentence embeddings for semantic similarity and ranking.
- **FAISS**: For approximate nearest neighbor vector search to efficiently retrieve relevant document sections (**faiss-cpu** for CPU-only environments).
- **Python standard libraries**: json, os, re, datetime, etc.
- **Additional libraries**: numpy, tqdm



## Input/Output Format

### Output JSON Structure
```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

## Prerequisites 

- [Docker](https://www.docker.com/get-started) installed on your machine (recommended for consistent environments)
- Python 3.8+ environment (if running locally)

---

## 🚀 Quick Start: Build & Run

```bash
# 1. Build the Docker image
docker build --platform linux/amd64 -t
mysolutionname:somerandomidentifier .

# 2. Place your test PDFs in the input/ directory
mkdir -p input output
cp yourfile.pdf input/

# 3. Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --
network none mysolutionname:somerandomidentifier`
```

> ✅ JSON output will appear in the `output/` folder.


---
