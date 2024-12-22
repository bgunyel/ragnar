import time
import random

import streamlit as st
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain


template = """
Your name is Ragnar. 

Answer the following question:

Here is the conversation history: {context}

Question: {question}

Answer:    
"""

model = ChatOllama(model='llama3:8b', temperature=0.5, num_predict=256)
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=256)
prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model

if __name__ == '__main__':

    st.set_page_config(
        page_title="Ragnar",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message['role'] != 'system':
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    user_message = st.chat_input("What is up?")
    if user_message:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_message)

        # Display assistant response in chat message container
        response = chain.stream({'context': st.session_state.messages, 'question': user_message})
        with st.chat_message("assistant"):
            result = st.write_stream(response)  ##
        st.session_state.messages.append({"role": "assistant", "content": result})
