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

• Retrieval simulation using **TF-IDF and Dense Retrieval (Sentence-BERT)**  
• Speaker fairness metrics for **representation imbalance**  
• Linguistic feature extraction for **communication style analysis**  
• Logistic regression models predicting **retrieval likelihood**  
• Visualization pipeline for analyzing **retrieval bias**

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
│   │   ├── participation_vs_retrieval_tfidf.png
│   │   ├── participation_vs_retrieval_dense.png
│   │   ├── speaker_representation_comparison.png
│   │   ├── retrieval_histogram_tfidf.png
│   │   ├── retrieval_histogram_dense.png
│   │   └── gini_comparison.png
│   ├── logs
│   └── results
│       ├── retrieval_results_tfidf.csv
│       ├── retrieval_results_dense.csv
│       ├── speaker_metrics_tfidf.csv
│       ├── speaker_metrics_dense.csv
│       ├── message_level_features_tfidf.csv
│       ├── message_level_features_dense.csv
│       ├── regression_coefficients_tfidf.csv
│       ├── regression_coefficients_dense.csv
│       ├── regression_coefficients_style_only_tfidf.csv
│       ├── regression_coefficients_style_only_dense.csv
│       ├── regression_summary_tfidf.txt
│       └── regression_summary_dense.txt
│
├── scripts
│   ├── prepare_dataset.py
│   ├── run_analysis.py
│   ├── run_pipeline.py
│   ├── run_retrieval.py
│   └── generate_plots.py
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
Two retrieval mechanisms simulate how AI assistants retrieve conversational messages:

• **TF-IDF retrieval** – lexical similarity baseline  
• **Dense retrieval (Sentence-BERT)** – semantic similarity retrieval

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

```bash
python3 scripts/prepare_dataset.py   --input data/raw/conversations.csv   --output data/processed/conversations_clean.csv
```

## Step 2 — Run Retrieval

TF-IDF:

```bash
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results.csv   --top-k 5   --retriever tfidf
```

Dense Retrieval:

```bash
python3 scripts/run_retrieval.py   --input data/processed/conversations_clean.csv   --output outputs/results/retrieval_results.csv   --top-k 5   --retriever dense
```

## Step 3 — Run Analysis

```bash
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_tfidf.csv   --output-dir outputs/results
```

```bash
python3 scripts/run_analysis.py   --messages data/processed/conversations_clean.csv   --retrieval outputs/results/retrieval_results_dense.csv   --output-dir outputs/results
```

## 4. Visualization

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

### Retrieval Distribution Histogram

Displays how retrieved messages are distributed across speakers.

• TF-IDF histogram  
• Dense histogram

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
