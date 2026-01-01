from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..utils.config import Config

def get_vectorstore():
    """Returns the LangChain Chroma vector store instance."""
    
    # Use the embedding model directly from Config
    embedding_function = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL
    )
    
    return Chroma(
        persist_directory=str(Config.CHROMA_DIR),
        embedding_function=embedding_function,
        collection_name="legal_documents"
    )