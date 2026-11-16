readme_content = """# 🐞 Semantic Duplicate Bug Report Detection using Transformer-Based NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Sentence-BERT](https://img.shields.io/badge/Model-Sentence--BERT%20%2B%20Cross--Encoder-orange)](https://www.sbert.net/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-FAISS%20IndexFlatIP-brightgreen)](https://faiss.ai/)
[![Frontend](https://img.shields.io/badge/App-Streamlit%20%26%20FastAPI-red)](https://streamlit.io/)

> **Course Project:** CSEG4134 — Computational Linguistics and Natural Language Processing  
> **Objective:** Automatically detect and rank duplicate software bug reports by combining **Transformer contextual embeddings (`Sentence-BERT`)**, **sub-millisecond vector search (`FAISS`)**, **deep pairwise reranking (`Cross-Encoder`)**, and **technical metadata fusion (`Error Signatures`, `Component`, `Version`)** via a **Supervised Learning-to-Rank (LTR) Meta-Classifier**.

---

## 🎯 1. Problem Statement & Motivation

In large software engineering platforms like **Jira**, **Bugzilla**, and **GitHub Issues**, thousands of bug reports are submitted daily. Frequently, multiple users or QA engineers report the **exact same software defect using completely different vocabulary**:
* **User A:** *"The application crashes while uploading a PDF."*
* **User B:** *"The software closes unexpectedly when attaching a document."*

Traditional lexical methods (**TF-IDF / BM25**) fail due to the **Vocabulary Mismatch Problem** when two reports share few exact words. Conversely, relying purely on general semantic similarity can cause false positives across unrelated software modules.

### ✨ Our Proposed Novelty
We solve both challenges through a **3-Stage Hybrid Transformer + Technical Metadata Pipeline**:
1. **Stage 1 (Hybrid Candidate Retrieval):** Combines **Sentence-BERT (`all-MiniLM-L6-v2`)** 384-dimensional dense embeddings indexed in **Meta's FAISS (`IndexFlatIP`)** (`~0.5 ms` latency) with **TF-IDF** bigrams to shortlist the Top-15 candidates.
2. **Stage 2 (Cross-Encoder Verification):** Applies **`cross-encoder/ms-marco-MiniLM-L-4-v2`** token-level cross-attention across `(Query, Candidate)` pairs.
3. **Stage 3 (Technical Metadata & Supervised LTR Fusion):** Extracts **Java Exception/Error Signatures** via Regex (`NullPointerException`, `CoreException`, etc.) and fuses **`Component`**, **`Version`**, and **`Extracted_Errors`** with NLP scores using a **5-Fold Cross-Validated Logistic Regression Meta-Classifier**.

---

## 🏗️ 2. System Architecture

```text
Incoming Bug Report (Title + Description + Component + Version)
       │
       ├──► Regex Error Extractor ──► Extracted_Errors (e.g., NullPointerException)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Fast Hybrid Retrieval (9,469 Historical Eclipse JDT Bugs)   │
│  • Bi-Encoder: Sentence-BERT (all-MiniLM-L6-v2) + FAISS IndexFlatIP  │
│  • Lexical: TF-IDF (Unigrams + Bigrams for exact method/bug IDs)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ Top-15 Shortlisted Candidates (~0.5 ms)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Deep Pairwise Verification                                  │
│  • Cross-Encoder (ms-marco-MiniLM-L-4-v2) Joint Self-Attention       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Technical Metadata Fusion & Learned Meta-Classifier (LTR)   │
│  • Features: [SBERT, TF-IDF, Cross-Encoder, Component, Version, Err] │
│  • Output: Calibrated Duplicate Probability (0-100%) + Explainable UI│
└──────────────────────────────────────────────────────────────────────┘
