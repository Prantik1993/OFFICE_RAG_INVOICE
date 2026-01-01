import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag.chain import build_rag_chain
from src.ingestion.ingest import ingest_documents


st.set_page_config(page_title="Legal Policy RAG", layout="wide")
st.title("⚖️ Legal Policy Chatbot (Industry Standard)")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Admin Controls")
    
    if st.button("🔄 Ingest Documents"):
        with st.spinner("Ingesting Documents with Parent Retrieval..."):
            ingest_documents()
        st.success("Ingestion Complete!")
        # Clear cache to force reload of the new DB
        if "chain" in st.session_state:
            del st.session_state["chain"]
        st.rerun()
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Chain ---
if "chain" not in st.session_state:
    with st.spinner("Loading BGE-M3 & RAG Chain..."):
        st.session_state.chain = build_rag_chain()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def get_langchain_history():
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

if prompt := st.chat_input("Ask about specific articles (e.g., 'What is Article 4?')"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        try:
            # Stream the response
            chunks = st.session_state.chain.stream({
                "input": prompt,
                "chat_history": get_langchain_history()
            })
            
            for chunk in chunks:
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")