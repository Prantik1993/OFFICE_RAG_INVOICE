import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

def ingest_documents():
    """
    Ingests PDFs using standard LangChain loaders.
    """
    print(f"📂 Scanning {Config.RAW_DOCS_DIR}...")
    
    # 1. Load Documents
    # DirectoryLoader with PyMuPDFLoader automatically handles opening PDFs
    # and extracting text + metadata (page numbers, filenames)
    loader = DirectoryLoader(
        str(Config.RAW_DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} pages.")

    # 2. Split Text
    # RecursiveCharacterTextSplitter is the standard for RAG.
    # It tries to split on paragraphs (\n\n) first, then sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks.")

    # 3. Index in Chroma
    # We use our helper from vectorstore.store to get the DB and add chunks
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()