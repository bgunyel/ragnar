import datetime
import os
import time
import traceback
import logging
import json
import streamlit as st

from config import settings
from ragnar.apps.fastapi_client import FastAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_stream_from_response(content: str):
    for chunk in content.split(sep=' '):
        chunk += ' '
        yield chunk
        time.sleep(0.05)


def _setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Ragnar - Business Intelligence Assistant (FastAPI)",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/bgunyel/ragnar',
            'Report a bug': 'https://github.com/bgunyel/ragnar/issues',
            'About': "Ragnar - Advanced Business Intelligence Assistant via FastAPI"
        }
    )


def _initialize_session_state(api_base_url: str):
    """Initialize all session state variables."""
    default_states = {
        "messages": [],
        "api_status": None,
        "api_error": None,
        "conversation_id": None,
        "processing": False,
        "last_response_time": None,
        "total_cost": 0.0,
        "conversation_started_at": datetime.datetime.now().astimezone(settings.TIME_ZONE),
        "api_base_url": api_base_url
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _render_api_status():
    """Render API status information."""
    if st.session_state.api_status:
        status_data = st.session_state.api_status
        health = status_data.get("health", {})
        detailed_status = status_data.get("status", {})

        # Overall status
        if health.get("status") == "healthy":
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")

        # Agent ready status
        agent_ready = health.get("agent_ready", False)
        if agent_ready:
            st.success("🤖 Agent Ready")
        else:
            st.warning("🤖 Agent Not Ready")

        # Detailed status
        if detailed_status:
            components = detailed_status.get("components", {})

            col1, col2 = st.columns(2)
            with col1:
                agent_status = components.get("agent", "unknown")
                if agent_status == "ready":
                    st.success(f"🤖 Agent: {agent_status}")
                else:
                    st.warning(f"🤖 Agent: {agent_status}")

            with col2:
                db_status = components.get("database", "unknown")
                if db_status == "connected":
                    st.success(f"🗄️ DB: {db_status}")
                else:
                    st.warning(f"🗄️ DB: {db_status}")

            # Models
            models = components.get("models", [])
            if models:
                st.caption(f"Models: {', '.join(models)}")

        # Timestamp info
        api_timestamp = health.get("timestamp")
        if api_timestamp:
            st.caption(f"API Time: {api_timestamp}")

        last_check = status_data.get("last_check")
        if last_check:
            st.caption(f"Last checked: {last_check.strftime('%H:%M:%S')}")

    elif st.session_state.api_error:
        st.error(f"🔴 Connection Error")
        with st.expander("Error Details", expanded=False):
            st.code(st.session_state.api_error, language="text")
    else:
        st.info("🔵 Not connected - Click 'Test Connection' to check API status")


def _render_session_metrics():
    """Render session metrics and statistics."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Messages", len(st.session_state.messages))
        if st.session_state.last_response_time:
            st.metric("Last Response", f"{st.session_state.last_response_time:.2f}s")

    with col2:
        st.metric("Total Cost", f"$ {st.session_state.total_cost:.4f}")
        session_duration = datetime.datetime.now().astimezone(settings.TIME_ZONE) - st.session_state.conversation_started_at
        st.metric("Session Duration", str(session_duration).split('.')[0])


def _clear_conversation():
    """Clear conversation history."""
    st.session_state.messages = []
    st.session_state.total_cost = 0.0
    st.session_state.conversation_started_at = datetime.datetime.now().astimezone(settings.TIME_ZONE)
    st.rerun()


def _export_conversation():
    """Export conversation to downloadable format."""
    if not st.session_state.messages:
        st.warning("No messages to export.")
        return

    export_data = {
        "conversation_id": st.session_state.conversation_id,
        "started_at": st.session_state.conversation_started_at.isoformat(),
        "exported_at": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat(),
        "api_backend": st.session_state.api_base_url,
        "messages": st.session_state.messages,
        "metrics": {
            "total_messages": len(st.session_state.messages),
            "total_cost": st.session_state.total_cost,
        }
    }

    json_str = json.dumps(export_data, indent=2, default=str)

    st.download_button(
        label="📄 Download JSON",
        data=json_str,
        file_name=f"ragnar_conversation_{datetime.datetime.now().astimezone(settings.TIME_ZONE).strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def _check_api_connection() -> bool:
    """Check if API is connected and ready."""
    if not st.session_state.api_status:
        return False

    status_data = st.session_state.api_status
    health = status_data.get("health", {})
    return health.get("status") == "healthy" and health.get("agent_ready", False)


class StreamlitFastAPIUI:
    """Streamlit UI that connects to FastAPI backend instead of direct agent usage."""

    def __init__(self, api_base_url: str = None):
        # Use environment variable or default
        if api_base_url is None:
            api_base_url = os.getenv('RAGNAR_API_URL', f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}")

        self.api_client = FastAPIClient(api_base_url)
        _setup_page_config()
        _initialize_session_state(api_base_url)
        self._setup_sidebar()

    def _setup_sidebar(self):
        """Setup sidebar with configuration options and metrics."""
        with st.sidebar:
            st.title("⚙️ Configuration")

            # API Configuration
            with st.expander("🔗 API Settings", expanded=True):
                self._render_api_settings()

            # API Status
            with st.expander("🔍 API Status", expanded=True):
                _render_api_status()

            # Session metrics
            with st.expander("📊 Session Metrics", expanded=True):
                _render_session_metrics()

            # Conversation management
            with st.expander("💬 Conversation", expanded=False):
                self._render_conversation_controls()

    def _render_api_settings(self):
        """Render API configuration controls."""
        new_url = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state.api_base_url,
            help="URL of your FastAPI backend (e.g., http://localhost:8000)"
        )

        if new_url != st.session_state.api_base_url:
            st.session_state.api_base_url = new_url
            self.api_client = FastAPIClient(new_url)
            st.session_state.api_status = None  # Reset status
            st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test Connection", type="primary", use_container_width=True):
                self._test_api_connection()

        with col2:
            if st.button("🔄 Auto Connect", type="secondary", use_container_width=True):
                self._auto_test_connection()

    def _auto_test_connection(self):
        """Automatically test connection without user interaction."""
        try:
            health = self.api_client.health_check()
            status = self.api_client.get_status()

            if health.get("status") == "healthy":
                st.session_state.api_status = {
                    "health": health,
                    "status": status,
                    "connected": True,
                    "last_check": datetime.datetime.now().astimezone(settings.TIME_ZONE)
                }
                st.session_state.api_error = None
            else:
                st.session_state.api_error = f"API returned unhealthy status: {health}"
                st.session_state.api_status = None
        except Exception as e:
            st.session_state.api_error = str(e)
            st.session_state.api_status = None

    def _test_api_connection(self):
        """Test connection to the FastAPI backend with user feedback."""
        with st.spinner("Testing API connection..."):
            self._auto_test_connection()

            if st.session_state.api_status:
                st.success("✅ Successfully connected to FastAPI backend!")
            else:
                st.error(f"❌ Connection failed: {st.session_state.api_error}")

    def _render_conversation_controls(self):
        """Render conversation management controls."""
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            _clear_conversation()

        if st.button("💾 Export Chat", type="secondary", use_container_width=True):
            _export_conversation()

        if st.button("🔄 Test API Again", type="secondary", use_container_width=True):
            self._test_api_connection()

    def _render_connection_required(self):
        """Render message when API connection is required."""
        st.warning("🔗 FastAPI Connection Required")
        st.markdown("""
        Please configure and test your FastAPI backend connection in the sidebar before chatting.

        **Steps:**
        1. Make sure your FastAPI backend is running:
           ```bash
           uv run python src/ragnar/apps/fastapi_app.py
           ```
           or
           ```bash
           uv run uvicorn src.ragnar.apps.fastapi_app:app --reload
           ```
        2. Check the API URL in the sidebar (default: http://localhost:8000)
        3. Click "Test Connection" to verify the connection
        4. Start chatting once connected!
        """)

        # Auto-test connection on first load
        if st.session_state.api_status is None:
            with st.spinner("Auto-testing connection..."):
                self._auto_test_connection()
                if st.session_state.api_status:
                    st.rerun()

    def _render_chat_interface(self):
        """Render the main chat interface."""
        # Chat header
        st.title("🔍 Ragnar - Business Intelligence Assistant")
        st.markdown(
            "*Connected to FastAPI Backend* | Ask me anything about business research, market analysis, or competitive intelligence!")

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
                            if "total_cost" in metadata:
                                st.metric("Message Token Cost", f"$ {metadata['total_cost']:.4f}")
                            elif "token_usage" in metadata:
                                st.caption(f"Message Token Usage: {metadata['token_usage']}")

        # Chat input with processing state
        if st.session_state.processing:
            st.chat_input("Processing your request...", disabled=True)
        else:
            user_message = st.chat_input("Ask Ragnar anything about business intelligence...")

            if user_message:
                self._process_user_message(user_message)

    def _process_user_message(self, user_message: str):
        """Process user message via FastAPI backend."""
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)

        # Set processing state
        st.session_state.processing = True

        try:
            # Process with FastAPI backend
            start_time = time.time()

            with st.chat_message("assistant"):
                with st.spinner("Analyzing your request via FastAPI..."):
                    # Use streaming response
                    response = self.api_client.send_message(message=user_message)
                    result = st.write_stream(stream=_make_stream_from_response(content=response['content']))

            end_time = time.time()
            response_time = end_time - start_time

            token_usage = response.get('token_usage', {})
            total_cost = response.get('total_cost', 0)
            cost_list = response.get('cost_list', [])

            # Create response metadata
            metadata = {
                "response_time": response_time,
                "timestamp": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat(),
                "api_backend": st.session_state.api_base_url,
                "token_usage": token_usage,
                "total_cost": total_cost,
                "cost_list": cost_list,
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
            st.session_state.total_cost += total_cost

        except Exception as e:
            error_msg = f"Error processing your request via FastAPI: {str(e)}"
            logger.error(f"Message processing failed: {e}")
            logger.error(traceback.format_exc())

            with st.chat_message("assistant"):
                st.error(error_msg)

                with st.expander("Error Details"):
                    st.code(traceback.format_exc(), language="text")
                    st.markdown("**Troubleshooting:**")
                    st.markdown("1. Check if FastAPI backend is running")
                    st.markdown("2. Verify API URL in sidebar")
                    st.markdown("3. Test connection using 'Test Connection' button")
                    st.markdown("4. Check FastAPI logs for errors")

            # Add error message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error**: {error_msg}",
                "metadata": {"error": True,
                             "timestamp": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat()}
            })

        finally:
            st.session_state.processing = False
            st.rerun()

    def render(self):
        """Main render method for the UI."""
        # Check API connection status
        if _check_api_connection():
            self._render_chat_interface()
        else:
            self._render_connection_required()


def main():
    """Main application entry point."""
    # Setup environment
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    try:
        # Get API URL from environment or use default
        api_url = os.getenv('RAGNAR_API_URL', f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}")

        ui = StreamlitFastAPIUI(api_base_url=api_url)
        ui.render()

    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        logger.error(f"App initialization failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()