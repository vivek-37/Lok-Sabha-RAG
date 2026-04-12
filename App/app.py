import streamlit as st
from rag_pipeline import retrieve_context
from llm import generate_answer, format_response

st.set_page_config(page_title="Lok Sabha RAG", page_icon="🏛️")

st.title("🏛️ Lok Sabha RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + Generating..."):

            context = retrieve_context(prompt)

            if not context:
                st.error("No relevant documents found.")
                st.stop()

            raw_answer = generate_answer(prompt, context)
            formatted = format_response(raw_answer)

            st.markdown(formatted)

            with st.expander("Context"):
                st.text(context)

            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted
            })