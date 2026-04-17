# Selective Memory: Measuring Retrieval Bias in AI-Augmented Group Conversations

**CS498 – Human-LLM Interaction**  
**Team:** Bhavyaa Chauhan, Jyoti Rawat, Sandra John  

---

# Overview

Selective Memory is a research project that investigates how AI retrieval systems select information from **group conversations** and whether certain speakers become disproportionately visible as a result.

Modern AI assistants increasingly rely on **retrieval-augmented generation (RAG)** to summarize discussions and answer questions. However, when these systems retrieve only a subset of messages, they may unintentionally **amplify some voices while suppressing others**.

This repository implements a **retrieval analysis pipeline** to measure whether AI retrieval mechanisms systematically favor certain speakers or linguistic styles.

---

# Key Features

• Retrieval simulation using **TF-IDF, Dense Retrieval (Sentence-BERT), and ChromaDB Vector Retrieval**  
• Speaker fairness metrics for **representation imbalance**  
• Linguistic feature extraction for **communication style analysis**  
• Logistic regression models predicting **retrieval likelihood**  
• Visualization pipeline for analyzing **retrieval bias**  
• Comparative analysis across **lexical retrieval, dense semantic retrieval, and vector database retrieval**
• Proposal-aligned **speaker-citing RAG summary generation** with JSONL trace logs  
• Human-study preparation artifacts for downstream perception evaluation

---

# Problem Statement

AI systems increasingly act as **computational memory** in collaborative environments such as:

• Slack / Discord conversations  
• Team meeting transcripts  
• Online discussion forums  
• Customer support interactions

When these systems retrieve only a subset of messages, they implicitly decide **which voices matter**.

This project explores whether:

• Some speakers are retrieved more often than others  
• Linguistic style influences retrieval likelihood  
• Retrieval bias may shape downstream summaries

---

# Research Questions

### RQ1 — Retrieval Fairness
Do retrieval systems disproportionately retrieve messages from certain speakers?

### RQ2 — Linguistic Influence
Do linguistic features (verbosity, hedging, questions, etc.) influence retrieval likelihood?

### RQ3 — Human Perception *(future work)*
Does selective retrieval influence how humans perceive speaker authority or expertise?

---

# Repository Structure

```
SP26-CS498-Selective-Memory
│
├── data
│   ├── raw
│   │   └── conversations.csv
│   └── processed
│       └── conversations_clean.csv
│
├── outputs
│   ├── figures
│   │   ├── participation_vs_retrieval_tfidf.png
│   │   ├── participation_vs_retrieval_dense.png
│   │   ├── participation_vs_retrieval_chroma.png
│   │   ├── speaker_representation_comparison.png
│   │   ├── retrieval_histogram_tfidf.png
│   │   ├── retrieval_histogram_dense.png
│   │   ├── retrieval_histogram_chroma.png
│   │   └── gini_comparison.png
│   ├── logs
│   │   ├── rag_trace_tfidf.jsonl
│   │   ├── rag_trace_dense.jsonl
│   │   └── rag_trace_chroma.jsonl
│   └── results
│       ├── retrieval_results_tfidf.csv
│       ├── retrieval_results_dense.csv
│       ├── retrieval_results_chroma.csv
│       ├── speaker_metrics_tfidf.csv
│       ├── speaker_metrics_dense.csv
│       ├── speaker_metrics_chroma.csv
│       ├── message_level_features_tfidf.csv
│       ├── message_level_features_dense.csv
│       ├── message_level_features_chroma.csv
│       ├── regression_coefficients_tfidf.csv
│       ├── regression_coefficients_dense.csv
│       ├── regression_coefficients_chroma.csv
│       ├── regression_coefficients_style_only_tfidf.csv
│       ├── regression_coefficients_style_only_dense.csv
│       ├── regression_coefficients_style_only_chroma.csv
│       ├── speaker_feature_summary_tfidf.csv
│       ├── speaker_feature_summary_dense.csv
│       ├── speaker_feature_summary_chroma.csv
│       ├── feature_retrieval_correlations_tfidf.csv
│       ├── feature_retrieval_correlations_dense.csv
│       ├── feature_retrieval_correlations_chroma.csv
│       ├── regression_summary_tfidf.txt
│       ├── regression_summary_dense.txt
│       ├── regression_summary_chroma.txt
│       ├── rag_summaries_tfidf.csv
│       ├── rag_summaries_dense.csv
│       ├── rag_summaries_chroma.csv
│       ├── human_study_template_tfidf.csv
│       └── retriever_comparison_summary.csv
│
├── scripts
│   ├── prepare_dataset.py
│   ├── run_analysis.py
│   ├── run_pipeline.py
│   ├── run_retrieval.py
│   ├── run_rag.py
│   ├── prepare_human_study.py
│   └── generate_plots.py
│
├── src/selective_memory
│   ├── features
│   │   └── linguistic_features.py
│   ├── metrics
│   │   ├── fairness_metrics.py
│   │   └── correlation_analysis.py
│   ├── models
│   │   └── logistic_regression.py
│   ├── rag
│   │   └── pipeline.py
│   ├── retrieval
│   │   ├── dense_retriever.py
│   │   ├── tfidf_retriever.py
│   │   └── chroma_retriever.py
│   └── utils
│       └── helpers.py
│
├── requirements.txt
└── README.md
```

---

# Pipeline Overview

The pipeline consists of five major stages:

### 1. Dataset Preparation
Raw conversation data is cleaned and normalized into a standardized format.

### 2. Retrieval Simulation
Three retrieval mechanisms simulate how AI assistants retrieve conversational messages:

• **TF-IDF retrieval** – lexical similarity baseline  
• **Dense retrieval (Sentence-BERT)** – semantic similarity retrieval  
• **ChromaDB retrieval** – vector database powered semantic retrieval

The ChromaDB implementation stores Sentence-BERT embeddings in a vector database and performs nearest-neighbor similarity search to retrieve the most relevant messages.

### 3. Retrieval Analysis
Speaker fairness metrics and logistic regression models analyze which messages are selected.

### 4. RAG Summary Generation
Retrieved messages can be turned into **speaker-citing summaries**. If `ANTHROPIC_API_KEY` and LangChain Anthropic dependencies are available, the pipeline uses Claude; otherwise it falls back to a deterministic extractive summarizer while still producing trace logs.

### 5. Visualization Pipeline
Generates plots to visualize speaker representation and retrieval bias across retrieval methods.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SP26-CS498-Selective-Memory.git
cd SP26-CS498-Selective-Memory
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python version recommended:

```
Python 3.9+
```

---

# Running the Pipeline

## Step 1 — Prepare Dataset

```bash
python3 scripts/prepare_dataset.py   --input data/raw/conversations.csv   --output data/processed/conversations_clean.csv
```

## Step 2 — Run Retrieval

TF-IDF:

```bash
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results_tfidf.csv   --top-k 5   --retriever tfidf
```

Dense Retrieval:

```bash
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results_dense.csv   --top-k 5   --retriever dense
```

ChromaDB Retrieval:

```bash
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results_chroma.csv   --top-k 5   --retriever chroma
```

## Step 3 — Run Analysis

```bash
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_tfidf.csv   --output-dir outputs/results
```

```bash
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_dense.csv   --output-dir outputs/results
```

```bash
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_chroma.csv   --output-dir outputs/results
```

## Step 4 — Generate Proposal-Style Summaries

```bash
python3 scripts/run_rag.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_chroma.csv   --output outputs/results/rag_summaries_chroma.csv   --trace-output outputs/logs/rag_trace_chroma.jsonl
```

## Step 5 — Prepare Human Study Template

```bash
python3 scripts/prepare_human_study.py   --summaries outputs/results/rag_summaries_chroma.csv   --output outputs/results/human_study_template_chroma.csv
```

## One-Command Pipeline

```bash
python3 scripts/run_pipeline.py   --input data/raw/conversations.csv   --top-k 5   --retrievers tfidf dense chroma   --run-generation
```

## Step 4 — Visualization

Generate all plots:

```
python3 scripts/generate_plots.py
```

Generated figures are stored in:

```
outputs/figures/
```

---

# Generated Visualizations

The visualization script generates several figures for analysis.

### Participation vs Retrieval Share

Shows how often each speaker participates versus how often they are retrieved.

• TF-IDF Retrieval  
• Dense Retrieval  
• ChromaDB Retrieval

### Retrieval Distribution Histogram

Displays how retrieved messages are distributed across speakers.

• TF-IDF histogram  
• Dense histogram  
• Chroma histogram

### Speaker Representation Ratio

Compares how speakers are over- or under-represented in retrieval results across retrieval methods.

### Gini Inequality Comparison

Compares inequality in retrieval attention across retrieval models.

---

# Fairness Metrics

### Participation Share

Fraction of total messages authored by a speaker.

### Retrieval Share

Fraction of retrieved messages attributed to that speaker.

### Representation Ratio

retrieval_share / participation_share

Interpretation:

1 → proportional representation  
>1 → over-represented  
<1 → under-represented  

### Gini Coefficient

Measures inequality in retrieval attention across speakers.

0 → perfectly equal distribution  
1 → maximum inequality  

---

# Logistic Regression Analysis

Two regression models are trained.

### Model 1 — Style + Speaker

Uses linguistic features and speaker identity.

### Model 2 — Style Only

Uses only linguistic features to measure the impact of communication style on retrieval likelihood.

---

# Dependencies

Core Python libraries:

```
pandas
numpy
scikit-learn
sentence-transformers
chromadb
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Results Summary

| Component | Status |
|-----------|--------|
Dataset preprocessing | Complete |
TF-IDF retrieval | Complete |
Dense retrieval | Complete |
ChromaDB retrieval | Complete |
Fairness metrics | Complete |
Linguistic feature extraction | Complete |
Regression analysis | Complete |
Visualization pipeline | Complete |
Human evaluation | Planned |

---

# Acknowledgments

Course: **CS498 – Human-LLM Interaction**  
Institution: **University of Illinois Urbana-Champaign**  
Semester: **Spring 2026**

---

# License

This repository is intended for **academic research and coursework**.
