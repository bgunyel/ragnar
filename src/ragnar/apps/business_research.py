import datetime
import os
import time
from typing import Any

import rich
import streamlit as st

from ai_common import LlmServers
from config import settings
from ragnar import BusinessIntelligenceAgent


def ui_app(llm_cfg: dict[str, Any]):

    bia = BusinessIntelligenceAgent(llm_config=llm_cfg,
                                    web_search_api_key=settings.TAVILY_API_KEY,
                                    database_url=settings.SUPABASE_URL,
                                    database_key=settings.SUPABASE_SECRET_KEY)

    ## UI Starts Here
    st.set_page_config(
        page_title="Ragnar",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_message = st.chat_input('Write to Ragnar')
    if user_message:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_message)

        # Display assistant response in chat message container
        response = bia.stream_response(user_message=user_message)

        with st.chat_message("assistant"):
            result = st.write_stream(response)  ##
        st.session_state.messages.append({"role": "assistant", "content": result})


if __name__ == '__main__':

    os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY
    os.environ['LANGSMITH_TRACING'] = settings.LANGSMITH_TRACING

    llm_config = {
        'language_model': {
            'model': 'llama-3.3-70b-versatile',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
            'model_args': {
                'service_tier': "auto",
                'temperature': 0,
                'max_retries': 5,
                'max_tokens': 32768,
                'model_kwargs': {
                    'top_p': 0.95,
                }
            }
        },
        'reasoning_model': {
            'model': 'qwen/qwen3-32b',  # 'deepseek-r1-distill-llama-70b',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
            'model_args': {
                'service_tier': "auto",
                'temperature': 0,
                'max_retries': 5,
                'max_tokens': 32768,
                'model_kwargs': {
                    'top_p': 0.95,
                }
            }
        }
    }

    ui_app(llm_cfg=llm_config)
