import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
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
        # Extract metadata like page numbers and source files
        source_info = f"Source: {meta.get('filename', 'doc')} (Page {meta.get('page_number', 0)})"
        formatted.append(f"{source_info}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain():
    """
    Builds the Modern Hybrid RAG Chain.
    Replaces: QA Pipeline, HybridRetriever, Reranker, and LLMClient.
    """
    
    # 1. Setup Semantic Search (Vector)
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K})

    # 2. Setup Keyword Search (BM25)
    # Note: efficient BM25 typically requires persisting the index. 
    # For simplicity here, we rebuild it from the vectorstore docs.
    # If your DB is huge, you should use a persistent BM25 retriever instead.
    try:
        # Fetch all docs to initialize BM25 (Warning: can be slow for large datasets)
        all_docs = vectorstore.get()["documents"]
        if all_docs:
            bm25_retriever = BM25Retriever.from_texts(all_docs)
            bm25_retriever.k = Config.TOP_K
            
            # Combine into Hybrid Search (Ensemble)
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
        else:
            base_retriever = vector_retriever
    except Exception as e:
        print(f"⚠️ BM25 Init failed: {e}. Falling back to standard vector search.")
        base_retriever = vector_retriever

    # 3. Setup Reranking (Cross Encoder)
    # This replaces your 'src/rag/reranker.py'
    try:
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=Config.TOP_K)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    except Exception:
        print("⚠️ Reranker model failed to load. Using base retriever.")
        final_retriever = base_retriever

    # 4. Define Prompt (Replaces prompt_templates.py)
    template = """You are a Legal Policy Assistant. Answer based ONLY on the context below.
    
    CRITICAL RULES:
    1. If the answer is not in the context, say "I don't have information about this in the available documents."
    2. Always cite the Source and Page Number provided in the context chunks.
    
    Context:
    {context}

    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Define LLM (Replaces llm_client.py)
    llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0)

    # 6. Build the Chain (LCEL)
    # This replaces 'qa_pipeline.py' manual stitching
    chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain