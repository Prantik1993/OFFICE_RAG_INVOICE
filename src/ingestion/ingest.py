import os
import shutil
import pickle
import logging
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

logger = logging.getLogger(__name__)

def ingest_documents():
    """
    Ingests PDFs with updated chunking settings.
    This will WIPE the existing database to ensure new settings are applied.
    """
    logger.info(f"Scanning {Config.RAW_DOCS_DIR}...")
    logger.info(f"Config: Chunk Size={Config.CHUNK_SIZE}, Overlap={Config.CHUNK_OVERLAP}")
    
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        is_separator_regex=True,
        separators=[
            r"\nArticle \d+",    # Splits on "Article 1", "Article 2"
            r"\n\d+\.\s+",       # Splits on "1. ", "2. " (Like Article 1.1)
            r"\n\(\w\)\s+",      # Splits on "(a)", "(b)" (Like Article 2.1.a)
            r"\n\n",             # Paragraph breaks
            r"\.",               # Sentences
            " "
        ]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks (Previous: ~{len(documents)*2}).")

    # 3. Clear Old Vector Store (Critical for applying new chunk sizes)
    if os.path.exists(Config.CHROMA_DIR):
        try:
            shutil.rmtree(Config.CHROMA_DIR)
            logger.info(f"CLEARED old database at {Config.CHROMA_DIR}")
        except Exception as e:
            logger.error(f"Failed to clear old vector store: {e}")

    # 4. Index Vectors
    logger.info("Indexing Vectors...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    # 5. Build BM25
    logger.info("Building BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.FETCH_K  # Allow BM25 to search broad too
    
    os.makedirs(os.path.dirname(Config.BM25_INDEX_PATH), exist_ok=True)
    with open(Config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    logger.info("Ingestion complete! Restart the Streamlit app now.")

if __name__ == "__main__":
    ingest_documents()