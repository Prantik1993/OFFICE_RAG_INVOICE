import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
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
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chain
if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def get_chat_history():
    """Convert Streamlit history to LangChain format."""
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

if prompt := st.chat_input("Ask about company policy..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        # Prepare inputs with history
        chat_history = get_chat_history()
        
        # Stream response
        # Note: chain now expects a dict with 'input' and 'chat_history'
        chunks = st.session_state.chain.stream({
            "input": prompt,
            "chat_history": chat_history
        })
        
        for chunk in chunks:
            full_response += chunk
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
    
    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})