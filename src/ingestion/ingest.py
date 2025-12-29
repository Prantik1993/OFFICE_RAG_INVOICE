import os
import pickle
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

def ingest_documents():
    """
    Ingests PDFs:
    1. Loads text from PDFs.
    2. Splits text using a hierarchy-aware splitter.
    3. Indexes vectors in ChromaDB.
    4. Builds and persists BM25 keyword index.
    """
    print(f" Scanning {Config.RAW_DOCS_DIR}...")
    
    # 1. Load Documents
    loader = DirectoryLoader(
        str(Config.RAW_DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Split Text (Optimized for Legal)
    # Tries to split by major structural elements first
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\nArticle ", "\nSection ", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Index in Chroma (Vector Store)
    print("Indexing Vectors...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    # 4. Build & Persist BM25 (Keyword Search)
    print("Building BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.TOP_K
    
    # Save BM25 index to disk to fix scalability issue
    with open(Config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"Ingestion complete. BM25 index saved to {Config.BM25_INDEX_PATH}")

if __name__ == "__main__":
    ingest_documents()