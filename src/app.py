from dotenv import load_dotenv
import streamlit as st

from rag.coordinator import retrieve_and_generate


# Load config
load_dotenv()

# App title
st.set_page_config(page_title="RAG-Powered Chatbot", initial_sidebar_state="collapsed")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant" and prompt is not None:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = retrieve_and_generate(st.session_state.messages, prompt)
                placeholder = st.empty()
                full_response = ""
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            except Exception as e:
                print(f"Error during retrieve_and_generate: {e}")
                full_response = "Error occurred, try again later."
                placeholder = st.empty()
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
