import os
import pickle
import logging
from typing import Optional, Any

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

logger = logging.getLogger(__name__)

def get_parent_document_retriever():
    """Builds the Vector Retriever (Parent-Child)."""
    vectorstore = get_vectorstore()
    
    fs = LocalFileStore(str(Config.DOC_STORE_DIR))
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    # Validation requires the splitter definition
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": Config.FETCH_K}
    )

def load_bm25_retriever() -> Optional[Any]:
    """Loads the Keyword Retriever."""
    if not os.path.exists(Config.BM25_INDEX_PATH):
        return None
    try:
        with open(Config.BM25_INDEX_PATH, "rb") as f:
            retriever = pickle.load(f)
            retriever.k = Config.FETCH_K
            return retriever
    except Exception:
        return None

def get_final_retriever():
    """Combines Vector + BM25 + Reranker into one powerful retriever."""
    
    # 1. Base Retrievers
    vector_retriever = get_parent_document_retriever()
    bm25_retriever = load_bm25_retriever()

    if bm25_retriever:
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
    else:
        base_retriever = vector_retriever

    # 2. Reranker (Filter)
    try:
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=Config.TOP_K)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        return final_retriever
    except Exception as e:
        logger.warning(f"Reranker failed: {e}. Using base retriever.")
        return base_retriever