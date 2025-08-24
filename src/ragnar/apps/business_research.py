import datetime
import os
import time
import traceback
from typing import Any, Dict, Optional
import logging
import streamlit as st

from config import settings
from ragnar import BusinessIntelligenceAgent, get_llm_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitBusinessUI:
    """Enhanced Streamlit UI for Business Intelligence Agent with improved UX and error handling."""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self._setup_page_config()
        self._initialize_session_state()
        self._setup_sidebar()

    # noinspection PyMethodMayBeStatic
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Ragnar - Business Intelligence Assistant",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/ragnar',
                'Report a bug': 'https://github.com/your-repo/ragnar/issues',
                'About': "Ragnar - Advanced Business Intelligence Assistant"
            }
        )

    def _initialize_session_state(self):
        """Initialize all session state variables."""
        default_states = {
            "messages": [],
            "agent": None,
            "agent_error": None,
            "conversation_id": None,
            "model_settings": self.llm_config.copy(),
            "processing": False,
            "last_response_time": None,
            "total_tokens_used": 0,
            "conversation_started_at": datetime.datetime.now(),
            "total_cost": 0.0,
        }

        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _setup_sidebar(self):
        """Setup sidebar with configuration options and metrics."""
        with st.sidebar:
            st.title("⚙️ Configuration")

            # Model configuration section
            # with st.expander("🤖 Model Settings", expanded=False):
                # self._render_model_settings()

            # Session metrics
            with st.expander("📊 Session Metrics", expanded=True):
                self._render_session_metrics()

            # Conversation management
            with st.expander("💬 Conversation", expanded=False):
                self._render_conversation_controls()

    # noinspection PyMethodMayBeStatic
    def _render_model_settings(self):
        """Render model configuration controls."""
        # Language Model Settings
        st.subheader("Language Model")
        current_model = st.session_state.model_settings['language_model']['model']

        model_options = [
            'llama-3.3-70b-versatile',
            'llama-3.1-70b-versatile',
            'mixtral-8x7b-32768',
            'gemma2-9b-it'
        ]

        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(current_model) if current_model in model_options else 0,
            key="language_model_select"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.model_settings['language_model']['model_args']['temperature']),
            step=0.1,
            key="temperature_slider"
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1000,
            max_value=32768,
            value=st.session_state.model_settings['language_model']['model_args']['max_tokens'],
            step=1000,
            key="max_tokens_input"
        )

        # Update model settings if changed
        if (selected_model != current_model or
                temperature != st.session_state.model_settings['language_model']['model_args']['temperature'] or
                max_tokens != st.session_state.model_settings['language_model']['model_args']['max_tokens']):
            st.session_state.model_settings['language_model']['model'] = selected_model
            st.session_state.model_settings['language_model']['model_args']['temperature'] = temperature
            st.session_state.model_settings['language_model']['model_args']['max_tokens'] = max_tokens
            st.session_state.agent = None  # Force agent recreation
            st.rerun()

    # noinspection PyMethodMayBeStatic
    def _render_session_metrics(self):
        """Render session metrics and statistics."""
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Messages", len(st.session_state.messages))
            if st.session_state.last_response_time:
                st.metric("Last Response", f"{st.session_state.last_response_time:.2f}s")

        with col2:
            st.metric("Total Cost", f"$ {st.session_state.total_cost:.2f}")
            session_duration = datetime.datetime.now() - st.session_state.conversation_started_at
            st.metric("Session Duration", str(session_duration).split('.')[0])

    def _render_conversation_controls(self):
        """Render conversation management controls."""
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            self._clear_conversation()

        if st.button("💾 Export Chat", type="secondary", use_container_width=True):
            self._export_conversation()

        if st.button("🔄 Reset Agent", type="secondary", use_container_width=True):
            st.session_state.agent = None
            st.session_state.agent_error = None
            st.rerun()

    # noinspection PyMethodMayBeStatic
    def _clear_conversation(self):
        """Clear conversation history."""
        st.session_state.messages = []
        st.session_state.total_tokens_used = 0
        st.session_state.conversation_started_at = datetime.datetime.now()
        st.rerun()

    # noinspection PyMethodMayBeStatic
    def _export_conversation(self):
        """Export conversation to downloadable format."""
        if not st.session_state.messages:
            st.warning("No messages to export.")
            return

        export_data = {
            "conversation_id": st.session_state.conversation_id,
            "started_at": st.session_state.conversation_started_at.isoformat(),
            "exported_at": datetime.datetime.now().isoformat(),
            "model_config": st.session_state.model_settings,
            "messages": st.session_state.messages,
            "metrics": {
                "total_messages": len(st.session_state.messages),
                "total_tokens": st.session_state.total_tokens_used,
            }
        }

        import json
        json_str = json.dumps(export_data, indent=2, default=str)

        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"ragnar_conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # noinspection PyMethodMayBeStatic
    def _get_or_create_agent(self) -> Optional[BusinessIntelligenceAgent]:
        """Get existing agent or create new one with error handling."""
        if st.session_state.agent is not None and st.session_state.agent_error is None:
            return st.session_state.agent

        try:
            with st.spinner("Initializing Business Intelligence Agent..."):
                agent = BusinessIntelligenceAgent(
                    llm_config=st.session_state.model_settings,
                    web_search_api_key=settings.TAVILY_API_KEY,
                    database_url=settings.SUPABASE_URL,
                    database_key=settings.SUPABASE_SECRET_KEY
                )
                st.session_state.agent = agent
                st.session_state.agent_error = None
                logger.info("Business Intelligence Agent initialized successfully")
                return agent

        except Exception as e:
            error_msg = f"Failed to initialize agent: {str(e)}"
            st.session_state.agent_error = error_msg
            logger.error(f"Agent initialization failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def _render_error_state(self, error: str):
        """Render error state with troubleshooting options."""
        st.error("🚨 Agent Initialization Failed")

        with st.expander("Error Details", expanded=True):
            st.code(error, language="text")

        st.markdown("### Troubleshooting")
        st.markdown("""
        1. **Check API Keys**: Ensure all required API keys are properly configured
        2. **Database Connection**: Verify Supabase URL and secret key
        3. **Model Availability**: Try switching to a different model
        4. **Network**: Check your internet connection
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Initialization", type="primary"):
                st.session_state.agent = None
                st.session_state.agent_error = None
                st.rerun()

        with col2:
            if st.button("🔧 Reset Configuration", type="secondary"):
                st.session_state.model_settings = self.llm_config.copy()
                st.session_state.agent = None
                st.session_state.agent_error = None
                st.rerun()

    def _render_chat_interface(self, agent: BusinessIntelligenceAgent):
        """Render the main chat interface."""
        # Chat header
        st.title("🔍 Ragnar - Business Intelligence Assistant")
        st.markdown("Ask me anything about business research, market analysis, or competitive intelligence!")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📊 Response Metadata", expanded=False):
                        metadata = message["metadata"]
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if "timestamp" in metadata:
                                st.caption(f"Generated at: {metadata['timestamp']}")
                        with col2:
                            if "response_time" in metadata:
                                st.metric("Response Time", f"{metadata['response_time']:.2f}s")
                        with col3:
                            if "cost" in metadata:
                                st.metric("Token Cost", f"$ {metadata["cost"]:.4f}")


        # Chat input with processing state
        if st.session_state.processing:
            st.chat_input("Processing your request...", disabled=True)
        else:
            user_message = st.chat_input("Ask Ragnar anything about business intelligence...")

            if user_message:
                self._process_user_message(user_message, agent)

    # noinspection PyMethodMayBeStatic
    def _process_user_message(self, user_message: str, agent: BusinessIntelligenceAgent):
        """Process user message with comprehensive error handling."""
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)

        # Set processing state
        st.session_state.processing = True

        try:
            # Process with agent
            start_time = time.time()

            with st.chat_message("assistant"):
                with st.spinner("Analyzing your request..."):
                    out_dict = agent.run(query=user_message)
                    response_stream = self.stream_response(text=out_dict['content'])
                    result = st.write_stream(response_stream)

            end_time = time.time()
            response_time = end_time - start_time

            # Create response metadata
            metadata = {
                "response_time": response_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "token_usage": out_dict['token_usage'],
                "cost": out_dict['total_cost']
            }

            # Add assistant response to history
            assistant_message = {
                "role": "assistant",
                "content": result,
                "metadata": metadata
            }
            st.session_state.messages.append(assistant_message)

            # Update session metrics
            st.session_state.last_response_time = response_time
            st.session_state.total_cost += out_dict['total_cost']
            st.session_state.total_tokens_used += 1  # Placeholder - would need actual token count

        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            logger.error(f"Message processing failed: {e}")
            logger.error(traceback.format_exc())

            with st.chat_message("assistant"):
                st.error(error_msg)

                with st.expander("Error Details"):
                    st.code(traceback.format_exc(), language="text")

            # Add error message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error**: {error_msg}",
                "metadata": {"error": True, "timestamp": datetime.datetime.now().isoformat()}
            })

        finally:
            st.session_state.processing = False
        
        # Rerun outside the finally block to avoid unreachable code warning
        st.rerun()

    # noinspection PyMethodMayBeStatic
    def stream_response(self, text: str):
        for chunk in text.split(sep=' '):
            chunk += ' '
            yield chunk
            time.sleep(0.05)

    def render(self):
        """Main render method for the UI."""
        # Try to get or create agent
        agent = self._get_or_create_agent()

        if st.session_state.agent_error:
            self._render_error_state(st.session_state.agent_error)
        elif agent is None:
            st.error("Unable to initialize agent. Please check the sidebar for configuration.")
        else:
            self._render_chat_interface(agent)


def create_llm_config() -> Dict[str, Any]:
    """Create LLM configuration with environment validation."""
    required_settings = ['GROQ_API_KEY', 'TAVILY_API_KEY', 'SUPABASE_URL', 'SUPABASE_SECRET_KEY']
    missing_settings = [setting for setting in required_settings if not hasattr(settings, setting)]

    if missing_settings:
        st.error(f"Missing required settings: {', '.join(missing_settings)}")
        st.stop()

    llm_config = get_llm_config()
    return llm_config


def main():
    """Main application entry point."""
    # Setup environment
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    try:
        llm_config = create_llm_config()
        ui = StreamlitBusinessUI(llm_config)
        ui.render()

    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        logger.error(f"App initialization failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()