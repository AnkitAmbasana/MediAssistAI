<h2 align="center">MediAssist AI: Evidence-Based Medical Query Assistant</h2>
<h2 align="center">🩺📚💊🧠</h2>

<p align="center"><b>Helping healthcare professionals make faster, research-backed clinical decisions with Gen AI + RAG</b></p>

<p align="center">
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/🖥️%20Streamlit-1.45.0-ff4b4b?logo=streamlit&logoColor=white"></a>
  <a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=white"></a>
  <a href="https://groq.com/"><img alt="Groq" src="https://img.shields.io/badge/⚡%20Groq-LLaMA3-blue?logo=groq&logoColor=white"></a>
  <a href="https://python.org/"><img alt="Python" src="https://img.shields.io/badge/🐍%20Python-3.13.0-3776AB?logo=python&logoColor=white"></a>
  <a href="https://chromadb.com/"><img alt="ChromaDB" src="https://img.shields.io/badge/📚%20ChromaDB-1.0.16-blue?logo=chromadb&logoColor=white"></a>
  <a href="https://python.org/dev/peps/pep-0566/"><img alt="python-dotenv" src="https://img.shields.io/badge/🐍%20python--dotenv-1.1.0-3776AB?logo=python&logoColor=white"></a>
  <a href="https://requests.readthedocs.io/"><img alt="Requests" src="https://img.shields.io/badge/🌐%20Requests-2.32.3-0052cc?logo=requests&logoColor=white"></a>
  <a href="https://www.sbert.net/"><img alt="Sentence Transformers" src="https://img.shields.io/badge/🧠%20Sentence%20Transformers-5.0.0-orange?logo=transformers&logoColor=white"></a>
</p>

---

## Overview

**MediInsight Health Solutions** is a US-based healthcare provider that helps doctors and research assistants deliver clear, evidence-based advice to patients.  
When a patient asks about a medical intervention — such as *intermittent fasting for Type 2 diabetes* — the scientific evidence can be scattered, inconclusive, or hard to retrieve quickly.

To solve this, we developed **MediAssist AI** — a **Retrieval-Augmented Generation (RAG)** system that automatically:
1. **Fetches medical research** from *PubMed* using APIs.
2. **Embeds & stores articles** in a vector database for fast semantic search.
3. **Retrieves relevant evidence** for a query.
4. **Generates clear, contextual answers** using the **Groq LLaMA 3 model**.

This empowers healthcare professionals to **save time, reduce uncertainty, and base recommendations on the latest peer-reviewed research**.

---

## Key Features

- **Automated PubMed Data Pipeline**:
  - Search and fetch the latest research articles for any medical topic.
  - Store articles in JSON format for reproducibility.
- **Vector Database with ChromaDB**:
  - Store embeddings for semantic similarity search.
  - Optimized for **biomedical text** with the **S-PubMedBert-MS-MARCO** model from Hugging Face.
- **RAG Pipeline with Groq LLaMA 3**:
  - Retrieve the most relevant research context.
  - Generate evidence-based responses.
- **Streamlit Web App**:
  - Sidebar to search & ingest articles.
  - Main area for natural language medical queries.
  - Option to manage (clear/delete) collections.
- **Modular Design**:
  - Easily swap embedding models or LLMs.
  - Environment variables stored securely in `.env`.

---


### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AnkitAmbasana/MediAssistAI.git
   cd MediAssistAI
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

## Contributing

To Contribute, please submit issues or pull requests for enhancements or fixes.

---

## License

Licensed under the Apache 2.0 License.

---
