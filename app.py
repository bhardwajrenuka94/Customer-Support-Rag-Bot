import streamlit as st
from rag_chain import chain

st.set_page_config(
    page_title="Customer Support Bot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Customer Support Bot")
st.caption("RAG · LangChain · ChromaDB · Gemini 1.5 Flash · HuggingFace")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your customer support assistant. Ask me about orders, refunds, payments, or account issues!"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            answer = chain.invoke(user_question)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })