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

logger = logging.getLogger(__name__)

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        file_path = meta.get("source", "Unknown")
        filename = os.path.basename(file_path) if file_path != "Unknown" else "doc"
        page = meta.get("page", 0) + 1 
        source_info = f"Source: {filename} (Page {page})"
        formatted.append(f"{source_info}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def load_bm25_retriever() -> Optional[Any]:
    if not os.path.exists(Config.BM25_INDEX_PATH):
        logger.warning("BM25 index not found. Run ingestion first.")
        return None
    try:
        with open(Config.BM25_INDEX_PATH, "rb") as f:
            retriever = pickle.load(f)
            # IMPORTANT: Set BM25 to fetch enough candidates
            retriever.k = Config.FETCH_K 
            return retriever
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        return None

def build_rag_chain() -> Any:
    """Builds the RAG Chain with optimized retrieval parameters."""
    
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)
    
    # --- 1. Retriever Setup ---
    vectorstore = get_vectorstore()
    
    # Use FETCH_K (100) to cast a wide net
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.FETCH_K})
    bm25_retriever = load_bm25_retriever()

    if bm25_retriever:
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    else:
        base_retriever = vector_retriever

    # --- 2. Reranker (The Filter) ---
    try:
        # The Cross-Encoder takes the 100 docs from base_retriever 
        # and picks the best TOP_K (10) specifically for the question.
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=Config.TOP_K)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    except Exception as e:
        logger.warning(f"Reranker failed: {e}. Falling back to base retriever.")
        final_retriever = base_retriever

    # --- 3. Prompts ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
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

    qa_system_prompt = """You are a Legal Policy Assistant. Answer based ONLY on the context below.
    
    CRITICAL RULES:
    1. If the answer is not in the context, say "I don't have information about this in the available documents."
    2. Always cite the Source and Page Number.
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # --- 4. Final Chain ---
    chain = (
        RunnablePassthrough.assign(
            context=RunnableBranch(
                (
                    lambda x: bool(x.get("chat_history")), 
                    history_aware_retriever | format_docs 
                ),
                (final_retriever | format_docs)
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain