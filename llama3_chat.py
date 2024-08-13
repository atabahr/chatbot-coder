import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from transformers import pipeline

import os
summarizer = pipeline("summarization")

st.subheader("Llama 3.1 405B Chatbot")

with st.sidebar:
    st.title("Llama 3.1 405B Chatbot")
    st.subheader("This app lets you chat with Llama 3.1 405B! [👉]")
    api_key = st.text_input("Enter your Fireworks API Key", type="password")
    add_vertical_space(2)
    st.markdown("""
    Want to learn how to build this? 
   
    Join [GenAI Course](https://www.buildfastwithai.com/genai-course) by Build Fast with AI!
    """)
    add_vertical_space(3)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/satvik-paramkusham)")

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Function to extract and summarize PDF
def extract_and_summarize_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Only initialize ChatOpenAI and ConversationChain if API key is provided
if api_key:
    if st.session_state.conversation is None:
        llm = ChatOpenAI(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            openai_api_key=api_key,
            openai_api_base="https://api.fireworks.ai/inference/v1"
        )
        st.session_state.conversation = ConversationChain(
            memory=st.session_state.buffer_memory, 
            llm=llm
        )

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Extracting and summarizing PDF..."):
            summary = extract_and_summarize_pdf(uploaded_file)
            st.write("Summary of PDF:")
            st.write(summary)

    # Rest of your chat interface code
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.predict(input=prompt)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
else:
    st.warning("Please enter your Fireworks API Key in the sidebar to start the chat.")