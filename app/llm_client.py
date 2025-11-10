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

            # Log what will be sent to the LLM for visibility/debugging
            try:
                logger.info(f"Prepared {len(messages)} user messages to send to LLM (including system prompt).")
                logger.debug(f"System prompt (truncated): {SYSTEM_PROMPT[:300].replace(chr(10), ' ')}")
                logger.debug(f"Conversation messages payload: {messages}")
            except Exception:
                # Non-fatal: continue even if logging fails
                logger.exception("Failed to log LLM payload")

            # Send request to LLM API with system prompt
            # Format for Ollama API
            payload = {
                "model": "llama2",  # or your chosen model
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages
                ],
                "stream": False
            }
            logger.debug(f"LLM request payload (truncated): {str(payload)[:800]}")

            response = await self.client.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()
            if "message" not in data:
                raise ValueError(f"Invalid response format from Ollama API: {data}")

            llm_response = data["message"]["content"]
            logger.info(f"Received response from LLM API: {llm_response[:100]}...")
            return llm_response

        except httpx.HTTPStatusError as e:
            # Log status code and response body for HTTP errors
            try:
                body = e.response.text
            except Exception:
                body = '<unable to read response body>'
            logger.error(f"HTTP error from LLM API: {e.response.status_code} - {body}")
            return None
        except httpx.RequestError as e:
            # Log full exception with traceback and request info for network errors
            logger.exception(f"Request error to LLM API: {e}")
            try:
                logger.debug(f"Failed request: {getattr(e, 'request', None)}")
            except Exception:
                pass
            return None
        except Exception as e:
            # Catch-all with traceback to help debugging unexpected issues
            logger.exception("Unexpected error calling LLM API")
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
