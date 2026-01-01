import os
import shutil
import pickle
import logging
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

# Initialize Logger
logger = logging.getLogger(__name__)

def ingest_documents():
    """
    Ingests PDFs:
    1. Loads text from PDFs.
    2. Splits text using a hierarchy-aware splitter (Regex enabled).
    3. Indexes vectors in ChromaDB (after clearing old data).
    4. Builds and persists BM25 keyword index.
    """
    logger.info(f"Scanning {Config.RAW_DOCS_DIR}...")
    
    # 1. Load Documents
    loader = DirectoryLoader(
        str(Config.RAW_DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages.")
    
    if not documents:
        logger.warning("No documents found. Exiting.")
        return

    # 2. Split Text (Optimized for Legal with Regex)
    # We enable 'is_separator_regex=True' to handle "1.", "2.", "a)" lists.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        is_separator_regex=True,
        separators=[
            r"\n\n",             # Double newlines (Paragraph breaks)
            r"\nArticle \d+",    # "Article 10", "Article 20"
            r"\nSection \d+",    # "Section 5"
            r"\n\d+\.\s+",       # Numbered lists: "1. ", "2. ", "10. "
            r"\n\(\d+\)\s+",     # Parenthesis numbers: "(1) ", "(2) "
            r"\n[a-z]\)\s+",     # Lettered lists: "a) ", "b) "
            r"\n",               # Standard line breaks
            r"\. ",              # Sentences
            " ",                 # Words
            ""                   # Characters
        ]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # 3. Index in Chroma (Vector Store)
    logger.info("Indexing Vectors...")
    
    # Clear existing vector database to prevent duplicates
    if os.path.exists(Config.CHROMA_DIR):
        try:
            shutil.rmtree(Config.CHROMA_DIR)
            logger.info(f"Cleared existing vector store at {Config.CHROMA_DIR}")
        except Exception as e:
            logger.error(f"Failed to clear old vector store: {e}")

    # Re-initialize vectorstore
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    # 4. Build & Persist BM25 (Keyword Search)
    logger.info("Building BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.TOP_K
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(Config.BM25_INDEX_PATH), exist_ok=True)
    
    # Save BM25 index to disk
    with open(Config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    logger.info(f"Ingestion complete. BM25 index saved to {Config.BM25_INDEX_PATH}")

if __name__ == "__main__":
    # Simple config for testing if run directly
    logging.basicConfig(level=logging.INFO)
    ingest_documents()