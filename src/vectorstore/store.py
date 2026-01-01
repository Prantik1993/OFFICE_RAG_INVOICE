from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..utils.config import Config

def get_embedding_function():
    """
    Returns the BAAI/bge-m3 embedding function.
    configured with normalization (crucial for BGE models).
    """
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vectorstore():
    """
    Returns the Chroma VectorStore configured for BGE-M3.
    Used by the ParentDocumentRetriever to store 'Child' chunks.
    """
    return Chroma(
        collection_name="split_parents", # Must match what ingest/chain expect
        persist_directory=str(Config.CHROMA_DIR),
        embedding_function=get_embedding_function()
    )