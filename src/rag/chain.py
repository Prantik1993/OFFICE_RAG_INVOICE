import os
import pickle
import logging
from typing import List, Any, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

# Initialize Logger
logger = logging.getLogger(__name__)

# ... imports ...

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        
        # FIX: PyMuPDFLoader uses 'source' and 'page'
        file_path = meta.get("source", "Unknown")
        
        # Extract just the filename (e.g., "policy.pdf") from the full path
        filename = os.path.basename(file_path) if file_path != "Unknown" else "doc"
        
        # Fix Page Number: PyMuPDF is 0-indexed, so we add 1 for humans
        page = meta.get("page", 0) + 1 
        
        source_info = f"Source: {filename} (Page {page})"
        formatted.append(f"{source_info}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)


def load_bm25_retriever() -> Optional[Any]:
    """Loads the persisted BM25 retriever or returns None on failure."""
    if not os.path.exists(Config.BM25_INDEX_PATH):
        logger.warning("BM25 index not found. Run ingestion first. Falling back to Vector only.")
        return None
    
    try:
        with open(Config.BM25_INDEX_PATH, "rb") as f:
            retriever = pickle.load(f)
            retriever.k = Config.TOP_K
            return retriever
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}. Falling back to Vector only.")
        return None

def build_rag_chain() -> Any:
    """
    Builds the Conversational RAG Chain.
    """
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)
    
    # --- 1. Retriever Setup ---
    vectorstore = get_vectorstore()
    fetch_k = getattr(Config, "FETCH_K", 20)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    bm25_retriever = load_bm25_retriever()

    if bm25_retriever:
        bm25_retriever.k = fetch_k
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    else:
        base_retriever = vector_retriever

    try:
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=Config.TOP_K)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    except Exception as e:
        logger.warning(f"Reranker model failed to load: {e}. Using base retriever.")
        final_retriever = base_retriever

    # --- 2. History-Awareness Logic ---
    
    # A. Contextualize Question Prompt
    # If the user says "What about specific cases?", this chain uses history 
    # to rewrite it to "What about specific cases of [Topic X]?"
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
        | final_retriever
    )

    # B. Answer Question Prompt
    qa_system_prompt = """You are a Legal Policy Assistant. Answer based ONLY on the context below.
    
    CRITICAL RULES:
    1. If the answer is not in the context, say "I don't have information about this in the available documents."
    2. Always cite the Source and Page Number provided in the context chunks.
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # --- 3. Build Final Chain ---
    
    # We define a branch:
    # - If chat_history is empty: pass 'input' directly to retriever
    # - If chat_history exists: pass through 'contextualize_q_prompt' first
    
    chain = (
        RunnablePassthrough.assign(
            context=RunnableBranch(
                (
                    lambda x: bool(x.get("chat_history")), 
                    history_aware_retriever | format_docs 
                ),
                (final_retriever | format_docs) # Fallback if no history
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain