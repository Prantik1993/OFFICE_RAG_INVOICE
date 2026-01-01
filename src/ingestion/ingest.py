import os
import re
import shutil
import pickle
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_community.retrievers import BM25Retriever
from ..vectorstore.store import get_vectorstore
from ..utils.config import Config
from ..utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def preprocess_legal_text(docs: List[Document]) -> List[Document]:
    """
    SMART CONTEXT INJECTION:
    Scans the documents for "Article X" headers.
    When it finds one, it prepends "Context: Article X - " to every subsequent 
    paragraph until it finds the next Article.
    
    This ensures that a chunk containing just "1. The regulation..." 
    becomes "Context: Article 1 - 1. The regulation...".
    """
    logger.info("Running Smart Legal Context Injection...")
    
    enriched_docs = []
    current_context = "General Preamble" # Default before Article 1
    
    # regex to find "Article 1", "Article 2", etc.
    # We look for "Article" followed by a number on its own line or start of line
    article_pattern = re.compile(r'(?:^|\n)(Article\s+\d+)', re.IGNORECASE)
    
    for doc in docs:
        content = doc.page_content
        
        # Split the page by lines to process linearly
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            # Check if this line defines a new Article
            match = article_pattern.search(line)
            if match:
                current_context = match.group(1).strip() # Update context to "Article 1"
            
            # Prepend context to the line if it's a substantive point (has text)
            if line.strip():
                # We add the context invisible to the user but visible to the embedding
                # format: [Article 1] The text...
                new_line = f"[{current_context}] {line}"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        # Reassemble the document
        enriched_text = "\n".join(new_lines)
        
        # Create new doc with modified content
        new_doc = Document(
            page_content=enriched_text,
            metadata=doc.metadata
        )
        enriched_docs.append(new_doc)
        
    return enriched_docs

def ingest_documents():
    logger.info(f"Scanning {Config.RAW_DOCS_DIR}...")
    
    # 1. Load PDFs
    loader = DirectoryLoader(
        str(Config.RAW_DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    raw_docs = loader.load()
    if not raw_docs:
        logger.warning("No documents found.")
        return
    logger.info(f"Loaded {len(raw_docs)} pages.")

    # 2. APPLY THE MAGIC FIX: Inject Context
    docs = preprocess_legal_text(raw_docs)

    # 3. Clear Old Stores
    if os.path.exists(Config.CHROMA_DIR):
        shutil.rmtree(Config.CHROMA_DIR)
    if os.path.exists(Config.DOC_STORE_DIR):
        shutil.rmtree(Config.DOC_STORE_DIR)

    # 4. Define Splitters
    # Parent: The full context is now even richer
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.PARENT_CHUNK_SIZE,
        chunk_overlap=Config.PARENT_CHUNK_OVERLAP
    )
    
    # Child: Smaller chunks, but now they ALL carry the "Article X" tag!
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        is_separator_regex=True,
        # We split on the injected context too
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
    ])

    # 5. Initialize Storage
    vectorstore = get_vectorstore()
    
    fs = LocalFileStore(str(Config.DOC_STORE_DIR))
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    # 6. Index
    logger.info("Indexing Enriched Chunks...")
    retriever.add_documents(docs, ids=None)
    
    # 7. Build BM25 (Hybrid Search)
    logger.info("Building BM25 Index...")
    parent_chunks = parent_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(parent_chunks)
    bm25_retriever.k = Config.FETCH_K
    
    os.makedirs(os.path.dirname(Config.BM25_INDEX_PATH), exist_ok=True)
    with open(Config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    logger.info("Ingestion Complete")

if __name__ == "__main__":
    ingest_documents()