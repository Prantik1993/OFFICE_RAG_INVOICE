# Legal RAG Chatbot ⚖️

A production-ready Retrieval-Augmented Generation (RAG) chatbot for querying company policy documents. Built with **LangChain**, **OpenAI**, **ChromaDB**, and **Streamlit**.

## ✨ Features

- 📄 **PDF Processing**: Extracts text, sections, and metadata from policy documents
- 🧠 **Semantic Search**: Uses HuggingFace embeddings for intelligent document retrieval
- 💾 **Persistent Storage**: Local ChromaDB vector store
- 🎯 **Confidence Scoring**: Evaluates answer quality with multiple metrics
- 📚 **Citation Tracking**: Provides section numbers and page references
- 🚦 **Guardrails**: Prevents hallucinations and ensures grounded answers
- 🖥️ **Interactive UI**: Clean Streamlit interface for testing

## 🏗️ Architecture

```
User Question
     ↓
[Retriever] → Search ChromaDB
     ↓
[Top-K Documents Retrieved]
     ↓
[LLM Client] → Generate Answer with Context
     ↓
[Confidence Scorer] → Evaluate Quality
     ↓
[Structured Answer] → Citations + Confidence
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenAI API key

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd legal-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 4. Add Your Documents

Place your PDF policy documents in `data/raw_docs/`:

```bash
data/raw_docs/
├── policy_1.pdf
├── policy_2.pdf
└── policy_3.pdf
```

### 5. Run the Application

```bash
# Start Streamlit UI
streamlit run src/app.py
```

Or use the command-line interface:

```python
from src.ingestion.ingest_pipeline import ingest_all_documents
from src.rag.qa_pipeline import ask_question

# Ingest documents
ingest_all_documents()

# Ask a question
answer = ask_question("What is the data retention policy?")
print(answer.to_display())
```

## 📖 Usage Guide

### Ingesting Documents

1. Place PDFs in `data/raw_docs/`
2. Click "Ingest Documents" in the sidebar
3. Wait for processing to complete

### Asking Questions

Example questions:
- *"What is the data retention policy?"*
- *"What are the requirements for data encryption?"*
- *"Who must I notify in case of a data breach?"*
- *"What is Section 7.2 about?"*

### Understanding Confidence Scores

- **High (≥80%)**: Answer is well-supported with strong citations
- **Medium (60-79%)**: Answer is reasonable but may lack some context
- **Low (<60%)**: Answer may be incomplete or uncertain

## 🔧 Advanced Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# Model Settings
OPENAI_MODEL=gpt-4              # or gpt-3.5-turbo
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Settings
CHUNK_SIZE=1000                 # Characters per chunk
CHUNK_OVERLAP=200              # Overlap between chunks
TOP_K_RESULTS=5                # Documents to retrieve
CONFIDENCE_THRESHOLD=0.7       # Minimum confidence to answer

# Logging
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

### Programmatic Usage

```python
from src.rag.qa_pipeline import QAPipeline
from src.vectorstore.chroma_manager import ChromaManager

# Initialize pipeline
pipeline = QAPipeline(use_llm_eval=True)  # Enable LLM-based evaluation

# Ask question
answer = pipeline.answer_question(
    question="What is the data classification policy?",
    top_k=5,
    min_confidence=0.7
)

# Access answer components
print(f"Answer: {answer.answer}")
print(f"Confidence: {answer.confidence:.2%}")
print(f"Citations: {[c.format() for c in answer.citations]}")

# Filter by section
answer = pipeline.answer_with_filter(
    question="What are the access controls?",
    section_number="7.2"
)
```

### Managing Documents

```python
from src.vectorstore.chroma_manager import ChromaManager

chroma = ChromaManager()

# List all documents
docs = chroma.list_documents()
print(f"Documents: {docs}")

# Get statistics
stats = chroma.get_collection_stats()
print(stats)

# Delete a document
chroma.delete_document("policy_1")

# Reset database (WARNING: deletes all data)
chroma.reset_collection()
```

## 📁 Project Structure

```
legal-rag-chatbot/
│
├── data/
│   ├── raw_docs/              # Place PDF files here
│   ├── chroma_db/             # Vector database (auto-created)
│   └── processed/             # Metadata backups
│
├── src/
│   ├── ingestion/             # PDF processing & chunking
│   ├── embeddings/            # HuggingFace embeddings
│   ├── vectorstore/           # ChromaDB manager
│   ├── rag/                   # RAG pipeline components
│   ├── models/                # Pydantic schemas
│   ├── utils/                 # Configuration & logging
│   └── app.py                 # Streamlit UI
│
├── logs/                      # Application logs
├── .env                       # Configuration (create from .env.example)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing

### CLI Testing

```bash
# Ingest documents
python -m src.ingestion.ingest_pipeline

# Test a question
python -c "from src.rag.qa_pipeline import ask_question; print(ask_question('What is Section 1 about?').to_display())"
```

### Interactive Testing

```bash
# Launch Streamlit UI
streamlit run src/app.py
```

## 🛠️ Troubleshooting

### Issue: "No PDF files found"

**Solution**: Ensure PDFs are in `data/raw_docs/` with `.pdf` extension

### Issue: "OpenAI API key not found"

**Solution**: Check `.env` file has `OPENAI_API_KEY=sk-...`

### Issue: "ImportError: No module named ..."

**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Low confidence answers

**Solutions**:
1. Increase `TOP_K_RESULTS` in `.env`
2. Reduce `CHUNK_SIZE` for better granularity
3. Enable LLM evaluation: `use_llm_eval=True`

## 🔐 Security Considerations

- **API Keys**: Never commit `.env` to version control
- **Data Privacy**: All processing is local except OpenAI API calls
- **Access Control**: Implement authentication for production use
- **Audit Logging**: All queries are logged to `logs/`

## 🚀 Production Deployment

### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add OpenAI API key in secrets
4. Deploy

### Option 2: Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/app.py", "--server.port=8501"]
```

### Option 3: FastAPI Backend

Convert to REST API:

```python
from fastapi import FastAPI
from src.rag.qa_pipeline import QAPipeline

app = FastAPI()
pipeline = QAPipeline()

@app.post("/ask")
async def ask_question(question: str):
    answer = pipeline.answer_question(question)
    return answer.dict()
```

## 📊 Performance Optimization

- **Embedding Cache**: First run downloads model (~100MB)
- **ChromaDB**: Indexed for fast retrieval
- **Batch Processing**: Use `ingest_directory()` for multiple files
- **GPU Support**: Enable for faster embeddings (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **OpenAI**: LLM API
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embeddings
- **Streamlit**: UI framework

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Check logs in `logs/` directory
- Review configuration in `.env`

---

**Built with ❤️ for better policy management**