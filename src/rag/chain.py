import os
import pickle
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from ..vectorstore.store import get_vectorstore
from ..utils.config import Config

def format_docs(docs):
    """Format retrieved documents for the prompt."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        source_info = f"Source: {meta.get('filename', 'doc')} (Page {meta.get('page_number', 0)})"
        formatted.append(f"{source_info}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def load_bm25_retriever():
    """Loads the persisted BM25 retriever or returns None on failure."""
    if not os.path.exists(Config.BM25_INDEX_PATH):
        print("⚠️ BM25 index not found. Run ingestion first. Falling back to Vector only.")
        return None
    
    try:
        with open(Config.BM25_INDEX_PATH, "rb") as f:
            retriever = pickle.load(f)
            retriever.k = Config.TOP_K
            return retriever
    except Exception as e:
        print(f"⚠️ Failed to load BM25 index: {e}. Falling back to Vector only.")
        return None

def build_rag_chain():
    """
    Builds the Conversational RAG Chain.
    """
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)
    
    # --- 1. Retriever Setup (Same as before) ---
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K})
    bm25_retriever = load_bm25_retriever()

    if bm25_retriever:
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
    except Exception:
        print("⚠️ Reranker model failed to load. Using base retriever.")
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
    
    def get_chat_history(x):
        return x.get("chat_history", [])

    # The Chain
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