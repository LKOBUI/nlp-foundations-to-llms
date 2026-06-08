# Hugging Face Transformers with Cosine Similarity

This directory contains practical implementations, notebooks, and exercises focused on mastering the **Hugging Face Ecosystem** for Natural Language Processing (NLP). It serves as a foundational step toward understanding and deploying Large Language Models (LLMs).

---

## Directory Structure

* **`HuggingFaceTransformers.ipynb`**: The primary Jupyter Notebook covering hands-on implementations of core Hugging Face components.


---

## Core Concepts Covered

The implementation in this repository guides you through the foundational workflow of using pre-trained models:

* **Pipeline API**: Quick and easy inference for tasks like sentiment analysis, text generation, and named entity recognition (NER).
* **Tokenizers**: Understanding how text is split into tokens, converted to input IDs, and prepared with attention masks for model consumption.
* **Models & Architectures**: Loading pre-trained weights from the Hugging Face Hub (e.g., BERT, GPT, RoBERTa) and understanding their underlying configurations.
* **Datasets**: Efficiently loading, processing, and batching text data using the `datasets` library.

## 📂 Contents
- `HuggingFaceTransformers.ipynb` → Jupyter notebook with step‑by‑step examples:
  - Loading pretrained models (`bert-base-uncased`, `distilbert-base-uncased`)
  - Tokenization and attention masks
  - Embedding extraction
  - **Cosine similarity for semantic similarity scoring**
  - Sentiment analysis pipeline
  - Fine‑tuning basics with Hugging Face Trainer API
---

## Setup and Installation

To run the notebook locally, follow these steps to set up your environment:

### 1. Clone the Repository
```bash
git clone 

```python
from transformers import pipeline

# Sentiment Analysis Example
classifier = pipeline("sentiment-analysis")
print(classifier("Transformers make NLP easy!"))

## Learning Objectives

Understand Hugging Face ecosystem (models, datasets, tokenizers).

Apply pretrained models for downstream tasks (classification, similarity).

Explore embeddings for semantic search and cosine similarity.

Prepare foundation for LLM fine‑tuning and deployment.