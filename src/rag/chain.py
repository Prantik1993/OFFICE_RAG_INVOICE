import os
import pickle
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    Builds the Modern Hybrid RAG Chain.
    Pipeline: Hybrid Search (Vector + BM25) -> Rerank (Cross-Encoder) -> LLM
    """
    
    # 1. Setup Semantic Search (Vector)
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K})

    # 2. Setup Keyword Search (BM25)
    bm25_retriever = load_bm25_retriever()

    if bm25_retriever:
        # Hybrid Search (Ensemble)
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    else:
        # Fallback to Vector only
        base_retriever = vector_retriever

    # 3. Setup Reranking (Cross Encoder)
    try:
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=Config.TOP_K)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    except Exception:
        print("⚠️ Reranker model failed to load. Using base retriever.")
        final_retriever = base_retriever

    # 4. Define Prompt
    template = """You are a Legal Policy Assistant. Answer based ONLY on the context below.
    
    CRITICAL RULES:
    1. If the answer is not in the context, say "I don't have information about this in the available documents."
    2. Always cite the Source and Page Number provided in the context chunks.
    
    Context:
    {context}

    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Define LLM
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)

    # 6. Build the Chain
    chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain