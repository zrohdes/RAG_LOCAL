import streamlit as st
from rag_chatbot import RAGChatbot
import os

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    chatbot = RAGChatbot(docs_dir="./docs")
    chatbot.setup()
    return chatbot

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG Chatbot")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        chatbot = get_chatbot()
        response = chatbot.query(prompt)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})