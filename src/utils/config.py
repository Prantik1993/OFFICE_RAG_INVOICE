import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 1. Configure Global Logging (Standard Practice)
# This setup ensures that logs from all files (ingest.py, chain.py) 
# share the same format and timestamp style.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    RAW_DOCS_DIR = DATA_DIR / "raw_docs"
    # New: Path to persist the BM25 index
    BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
    
    # Models
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # RAG Settings
    CHUNK_SIZE =4000
    CHUNK_OVERLAP = 500
    TOP_K = 10
    FETCH_K=20
    
    # LangSmith (Observability)
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "legal-policy-rag"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")