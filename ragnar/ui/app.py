import streamlit as st
from ragnar.backend.rag_engine import RagEngine


if __name__ == '__main__':

    rag_engine = RagEngine()

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
        response = rag_engine.get_response(user_message=user_message)
        with st.chat_message("assistant"):
            result = st.write_stream(response)  ##
        st.session_state.messages.append({"role": "assistant", "content": result})
