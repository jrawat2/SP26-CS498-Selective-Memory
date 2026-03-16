# Selective Memory: Measuring Retrieval Bias in AI-Augmented Group Conversations

**CS498 – Human-LLM interaction**  
**Team:** Bhavyaa Chauhan, Jyoti Rawat, Sandra John  

---

# Overview

Selective Memory is a research project that investigates how AI retrieval systems select information from **group conversations** and whether certain speakers become disproportionately visible as a result.

Modern AI assistants increasingly rely on **retrieval‑augmented generation (RAG)** to summarize discussions and answer questions. However, when these systems retrieve only a subset of messages, they may unintentionally **amplify some voices while suppressing others**.

This repository implements a **retrieval analysis pipeline** to measure whether AI retrieval mechanisms systematically favor certain speakers or linguistic styles.

---

# Key Features

• Conversation retrieval simulation using TF‑IDF  
• Speaker fairness metrics for representation imbalance  
• Linguistic feature extraction for communication style analysis  
• Logistic regression models predicting retrieval likelihood  
• Diagnostic outputs for analyzing retrieval bias

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
├── docs
│
├── outputs
│   ├── figures
│   ├── logs
│   └── results
│       ├── message_level_features.csv
│       ├── regression_coefficients_style_only.csv
│       ├── regression_coefficients.csv
│       ├── regression_summary.txt
│       ├── retrieval_results.csv
│       └── speaker_metrics.csv
│
├── scripts
│   ├── prepare_dataset.py
│   ├── run_analysis.py
│   ├── run_pipeline.py
│   └── run_retrieval.py
│
├── src/selective_memory
│   ├── features
│   │   └── linguistic_features.py
│   ├── metrics
│   │   └── fairness_metrics.py
│   ├── models
│   │   └── logistic_regression.py
│   ├── retrieval
│   │   ├── dense_retriever.py
│   │   └── tfidf_retriever.py
│   └── utils
│       └── helpers.py
│
├── tests
├── requirements.txt
└── README.md
```

---

# Pipeline Overview

The pipeline consists of three major stages:

### 1. Dataset Preparation
Raw conversation data is cleaned and normalized into a standardized format.

### 2. Retrieval Simulation
A TF‑IDF retriever simulates how AI systems retrieve relevant messages for predefined conversation queries.

### 3. Retrieval Analysis
Speaker fairness metrics and logistic regression models analyze which messages are selected.

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

```
python3 scripts/prepare_dataset.py   --input data/raw/conversations.csv   --output data/processed/conversations_clean.csv
```

---

## Step 2 — Run Retrieval

```
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results.csv   --top-k 5
```

---

## Step 3 — Run Analysis

```
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results.csv   --output-dir outputs/results
```

---

# Linguistic Features

The analysis extracts message-level communication features including:

• Message length (characters and words)  
• Presence of questions  
• Presence of exclamation marks  
• Uppercase character usage  
• Digit usage  
• Hedge word frequency  
• Politeness markers  
• First‑person pronoun usage  

---

# Fairness Metrics

### Participation Share
Fraction of total messages authored by a speaker.

### Retrieval Share
Fraction of retrieved messages attributed to that speaker.

### Representation Ratio

```
retrieval_share / participation_share
```

Interpretation:

```
1.0  → proportional representation
>1.0 → over‑represented
<1.0 → under‑represented
```

### Gini Coefficient

Measures inequality in retrieval attention across speakers.

```
0 → perfectly equal distribution
1 → maximum inequality
```

---

# Logistic Regression Analysis

Two models are implemented:

### Model 1 — Style + Speaker

Uses:

• Linguistic features  
• Speaker identity  

Purpose: determine whether linguistic features remain predictive after controlling for speaker identity.

---

### Model 2 — Style Only

Uses **only linguistic features**.

Purpose: determine whether communication style alone influences retrieval likelihood.

---

# Output Files

Results are written to:

```
outputs/results/
```

Generated files:

• retrieval_results.csv  
• speaker_metrics.csv  
• message_level_features.csv  
• regression_summary.txt  
• regression_coefficients.csv  
• regression_coefficients_style_only.csv  

---

# Dependencies

Core Python libraries:

```
pandas
numpy
scikit-learn
sentence-transformers
```

Install using:

```
pip install -r requirements.txt
```

---

# Testing

Run tests:

```
pytest tests/
```

---

# Future Work

• Sentence‑BERT dense retrieval  
• ChromaDB vector database indexing  
• LangChain‑based RAG pipeline  
• Claude LLM integration  
• LangSmith tracing  
• Human perception experiments

---

# Results Summary

| Component | Status |
|-----------|--------|
Dataset preprocessing | Complete |
TF‑IDF retrieval | Complete |
Fairness metrics | Complete |
Linguistic feature extraction | Complete |
Regression analysis | Complete |
Dense retrieval | Planned |
Human evaluation | Planned |

---

# Acknowledgments

Course: **CS498 – Human-LLM interaction**  
Institution: **University of Illinois Urbana‑Champaign**  
Semester: **Spring 2026**

---

# License

This repository is intended for academic research and coursework.
