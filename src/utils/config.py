import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    DOC_STORE_DIR = DATA_DIR / "doc_store"
    RAW_DOCS_DIR = DATA_DIR / "raw_docs"
    BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
    
    # --- Models ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # SETTING: BAAI/bge-m3
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DEVICE = "cuda"
    
    # --- Retrieval Settings (Parent-Child) ---
    PARENT_CHUNK_SIZE = 2000
    PARENT_CHUNK_OVERLAP = 200
    
    CHILD_CHUNK_SIZE = 400
    CHILD_CHUNK_OVERLAP = 50
    
    FETCH_K = 20 
    TOP_K = 5    
    
    # --- Observability ---
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "legal-policy-rag"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")