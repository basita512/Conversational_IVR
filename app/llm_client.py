import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_url=None, timeout=None):
        """Initialize LLM client.

        Args:
            api_url: Optional override for LLM API URL
            timeout: Optional override for LLM API timeout
        """
        self.api_url = api_url or settings.llm_api_url
        self.timeout = timeout or settings.llm_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"LLMClient initialized with API URL: {self.api_url}")
    
    async def get_response(self, conversation_history):
        """Get response from LLM API, including system prompt as context."""
        from app.config import SYSTEM_PROMPT
        logger.info("Sending conversation history and system prompt to LLM API...")

        try:
            # Format the messages for the API
            messages = []
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Send request to LLM API with system prompt
            response = await self.client.post(
                self.api_url,
                json={
                    "system_prompt": SYSTEM_PROMPT,
                    "messages": messages
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()
            if "response" not in data:
                raise ValueError(f"Invalid response format from LLM API: {data}")

            llm_response = data["response"]
            logger.info(f"Received response from LLM API: {llm_response[:100]}...")
            return llm_response

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from LLM API: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error to LLM API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling LLM API: {e}")
            return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info("LLMClient connection closed")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
