import streamlit as st
from src.rag.chain import build_rag_chain
from src.ingestion.ingest import ingest_documents

st.set_page_config(page_title="Legal Policy RAG", layout="wide")
st.title("⚖️ Legal Policy Chatbot (Modern Stack)")

# Sidebar
with st.sidebar:
    if st.button("🔄 Ingest Documents"):
        with st.spinner("Ingesting..."):
            ingest_documents()
        st.success("Done!")

# Initialize Chain
if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about company policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        # Stream response
        chunks = st.session_state.chain.stream(prompt)
        for chunk in chunks:
            full_response += chunk
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})