# Learning What to Crawl: Metadata-Enriched LLMs for Crawling Frontier Prioritisation

This repository contains the official implementation of the experiments presented in the MSc thesis:

> **Learning What to Crawl: Metadata-Enriched LLMs for Crawling Frontier Prioritisation**  
> Niccolò Settimelli, MSc in Artificial Intelligence and Data Engineering, University of Pisa (2025)  
> Supervisors: Prof. Nicola Tonellotto, Dott.ssa Francesca Pezzuti

---

## 🎯 Research Motivation and Objectives
Web crawling is a cornerstone of modern search engines. Classical frontier prioritisation strategies (e.g., PageRank) rely on **global graph analysis**, which is computationally expensive and unsuitable for real-time crawling.  
This thesis proposes a paradigm shift: **semantic-aware frontier prioritisation** based on transformer models enriched with metadata.  
The main contributions are:
- Moving from **page-centric quality estimation** to **outlink utility prediction**.  
- Designing and fine-tuning **LLM-based models** for binary classification and score estimation.  
- Extending models with **metadata fusion** (anchors, domains, numeric link features).  
- Integrating predictions into a **crawler simulation framework** and benchmarking against BFS, DFS, and PageRank.

---

## ❓ Research Questions
The repository addresses the following RQs from the thesis:

- **RQ1**: Can we train effective classifiers that predict whether a page contains semantically relevant outlinks?  
- **RQ2**: Can we produce continuous priority scores suitable for crawler frontier management?  
- **RQ3**: Does metadata integration (anchors, domains, numeric features) improve performance over text-only baselines?  
- **RQ4**: Do these models, once integrated into a real crawling pipeline, outperform classical baselines such as PageRank in practice?

---

## 📂 Repository Structure
The code is organised by experimental phase. For a detailed description of structure and reproducibility guidelines, **see Chapter 7 (“Code and Implementation”) of the thesis**.

