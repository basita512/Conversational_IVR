import logging
import httpx
from app.config import settings
from sentence_transformers import SentenceTransformer
import chromadb
import asyncio

logger = logging.getLogger(__name__)

# Load embedding model + ChromaDB at startup (global cache)
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./zenius_db")
    # collection = client.get_or_create_collection("zenius_kb")
    collection = client.get_collection("zenius_kb")
    logger.info("? Zenius KB successfully loaded into memory for RAG retrieval.")
except Exception as e:
    logger.exception("? Failed to initialize ChromaDB or embedding model.")


class LLMClient:
    def __init__(self, api_url=None, timeout=None):
        """Initialize LLM client."""
        self.api_url = api_url or settings.llm_api_url
        self.timeout = timeout or settings.llm_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"LLMClient initialized with API URL: {self.api_url}")

    async def _retrieve_context(self, user_query: str, n_results: int = 3) -> str:
        """Retrieve top relevant KB chunks using ChromaDB for contextual augmentation."""
        try:
            logger.info(f"?? Retrieving KB context for query: {user_query}")
            query_embedding = model.encode(user_query)
            
            # ? Convert ndarray to list
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

            if not results or not results["documents"] or not results["documents"][0]:
                logger.warning("?? No KB context found for this query.")
                return ""

            docs = results["documents"][0]
            context = "\n".join(docs)
            logger.debug(f"Retrieved {len(docs)} KB chunks for context.")
            return context
        except Exception as e:
            logger.exception("Error retrieving KB context.")
            return ""

    async def get_response(self, conversation_history):
        """Get response from LLM API, injecting Zenius KB context dynamically."""
        try:
            # Extract latest user message for retrieval
            user_query = ""
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break

            # 1?? Fetch relevant KB context from Chroma
            kb_context = await self._retrieve_context(user_query)

                    

            # 2?? Prepare messages for LLM API (system prompt now inside model)
            messages = []
            for i, msg in enumerate(conversation_history):
                content = msg["content"]
                
                # Inject KB context into the LAST user message (most recent query)
                if msg["role"] == "user" and content == user_query and kb_context:
                    content = f"[Knowledge Base Context: {kb_context}]\n\nUser query: {content}"
                    logger.info(f"💉 Injected KB context into user message")
                
                messages.append({
                    "role": msg["role"],
                    "content": content
                })

            # Log final messages being sent
            logger.info(f"📤 Sending {len(messages)} messages to LLM")
            for i, msg in enumerate(messages, 1):
                preview = msg['content'][:150] + '...' if len(msg['content']) > 150 else msg['content']
                logger.info(f"   {i}. {msg['role'].upper()}: {preview}")

            # for msg in conversation_history:
            #     messages.append({
            #         "role": msg["role"],
            #         "content": msg["content"]
            #     })

            # # Inject KB context at the start of conversation
            # if kb_context:
            #     messages.insert(0, {
            #         "role": "system",
            #         "content": (
            #             # f"Zenius Knowledge Base Context:\n{kb_context}\n\n"
            #             # "You are Zenius VoiceIQ assistant. "
            #             # "Respond concisely (one to two short sentences only). "
            #             # "Avoid repetition and unnecessary detail."
            #             f"Zenius Knowledge Base Context:\n{kb_context}\n\n"
            #             "You represent Zenius VoiceIQ, an intelligent enterprise voice platform. "
            #             "Always answer as the company ('we'), not as 'I'. "
            #             "Keep your tone professional and concise (one short sentence). "
            #             "Do not refer to yourself as an assistant."
            #         )
            #     })

            payload = {
                "model": "zenius-llm",  # custom Ollama model with built-in system prompt
                "messages": messages,
                "stream": False
            }

            logger.debug(f"LLM payload (truncated): {str(payload)[:800]}")

            # 3?? Send to Ollama API
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
            logger.info(f"? Received response from LLM API: {llm_response[:100]}...")
            return llm_response

        except httpx.HTTPStatusError as e:
            body = e.response.text if hasattr(e.response, "text") else "<unreadable>"
            logger.error(f"HTTP error from LLM API: {e.response.status_code} - {body}")
            return None
        except httpx.RequestError as e:
            logger.exception(f"Request error to LLM API: {e}")
            return None
        except Exception:
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
