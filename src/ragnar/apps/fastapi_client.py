import logging
from typing import Any, Dict

import requests
import rich

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastAPIClient:
    """Client to interact with the FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 300 second timeout

    def health_check(self) -> Dict[str, Any]:
        """Check if the FastAPI backend is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status from the FastAPI backend."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message and get the complete response."""
        try:
            payload = {"message": message}
            response = self.session.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            print("PRINTING RESPONSE.JSON:")
            rich.print(response.json())
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Send message {message} failed: {e}")
            raise Exception(f"API Error: {str(e)}")
