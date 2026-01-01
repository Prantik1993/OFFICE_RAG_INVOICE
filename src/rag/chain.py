import os
import logging
from typing import List, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# IMPORT FROM OUR NEW MODULES
from .retriever import get_final_retriever
from .prompts import get_contextualize_prompt, get_qa_prompt
from ..utils.config import Config

logger = logging.getLogger(__name__)

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        meta = doc.metadata
        file_path = meta.get("source", "Unknown")
        filename = os.path.basename(file_path) if file_path != "Unknown" else "doc"
        page = meta.get("page", 0) + 1 
        formatted.append(f"Source: {filename} (Page {page})\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain() -> Any:
    """Builds the final RAG chain by assembling modular components."""
    
    # 1. Setup Components
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)
    retriever = get_final_retriever()
    
    # 2. History Awareness Branch
    history_aware_retriever = (
        get_contextualize_prompt()
        | llm
        | StrOutputParser()
        | retriever
    )

    # 3. Main Chain Assembly
    chain = (
        RunnablePassthrough.assign(
            context=RunnableBranch(
                (lambda x: bool(x.get("chat_history")), history_aware_retriever | format_docs),
                (retriever | format_docs)
            )
        )
        | get_qa_prompt()
        | llm
        | StrOutputParser()
    )
    
    return chain