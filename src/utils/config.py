import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure Global Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    RAW_DOCS_DIR = DATA_DIR / "raw_docs"
    BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
    
    # --- Models ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Using MiniLM? It has a max token limit of 512 (~2000 chars).
    # Keep chunks well below this limit to ensure high-quality embeddings.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # --- RAG Tuning (FIX FOR ARTICLE 1) ---
    # 1. Smaller Chunks: Ensures "Article 1" is the main topic of the chunk, not buried.
    CHUNK_SIZE = 1000  # Reduced from 4000
    CHUNK_OVERLAP = 200 # Reduced from 500
    
    # 2. Broader Retrieval: Fetch more candidates (100) to find the "needle in the haystack",
    # then let the Reranker filter down to the best 10.
    FETCH_K = 100      # Increased from 20 (Critical for legal docs with many references)
    TOP_K = 10         # Final number of docs sent to LLM
    
    # --- Observability ---
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "legal-policy-rag"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")