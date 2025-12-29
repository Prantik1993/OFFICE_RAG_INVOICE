# Legal RAG Chatbot ⚖️

A production-ready Retrieval-Augmented Generation (RAG) chatbot for querying company policy documents. Built with **LangChain**, **OpenAI**, **ChromaDB**, and **Streamlit**.

## ✨ Features

- 🧠 **Hybrid Search**: Combines **Semantic Search** (Vectors) with **Keyword Search** (BM25) for high accuracy.
- 🎯 **Reranking**: Uses a Cross-Encoder to re-order results, ensuring the most relevant legal clauses appear first.
- 🚀 **Scalable**: Persists search indexes to disk to handle large document sets without memory crashes.
- 📄 **PDF Processing**: Automatic text extraction and metadata tracking (page numbers).
- 🚦 **Grounded Answers**: Strictly answers from context to prevent hallucinations.

## 🏗️ Architecture