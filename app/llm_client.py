import logging
import httpx
from app.config import settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import asyncio

logger = logging.getLogger(__name__)

# Load embedding model + ChromaDB at startup (global cache)
try:
    # Embedding model for initial retrieval
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    # Cross-encoder model for re-ranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    client = chromadb.PersistentClient(path="./Data/zenius_db")
    collection = client.get_collection("zenius_kb")
    logger.info("? Zenius KB successfully loaded into memory for RAG retrieval.")
    logger.info("? Re-ranker model loaded successfully.")
except Exception as e:
    logger.exception("? Failed to initialize ChromaDB or embedding model.")


class LLMClient:
    def __init__(self, api_url=None, timeout=None):
        """Initialize LLM client."""
        self.api_url = api_url or settings.llm_api_url
        self.timeout = timeout or settings.llm_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"LLMClient initialized with API URL: {self.api_url}")

    async def _retrieve_context(self, user_query: str, initial_results: int = 10, final_results: int = 1) -> str:
        """
        Retrieve top relevant KB chunks using two-stage retrieval:
        1. Retrieve top-N chunks using semantic similarity (embedding)
        2. Re-rank using cross-encoder and return top-K
        
        Args:
            user_query: The user's query string
            initial_results: Number of chunks to retrieve initially (default: 10)
            final_results: Number of top chunks to return after re-ranking (default: 1)
        
        Returns:
            Concatenated context string from top re-ranked chunks
        """
        try:
            logger.info(f"?? Retrieving KB context for query: {user_query}")
            
            # Stage 1: Initial retrieval using embeddings (top-10)
            query_embedding = model.encode(user_query)
            
            # Convert ndarray to list
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            results = collection.query(
                query_embeddings=[query_embedding], 
                n_results=initial_results
            )

            if not results or not results["documents"] or not results["documents"][0]:
                logger.warning("?? No KB context found for this query.")
                return ""

            docs = results["documents"][0]
            logger.info(f"?? Retrieved {len(docs)} initial KB chunks")
            
            # Stage 2: Re-ranking using cross-encoder
            if len(docs) > 0:
                # Create query-document pairs for re-ranking
                pairs = [[user_query, doc] for doc in docs]
                
                # Get re-ranking scores
                scores = reranker.predict(pairs)
                
                # Sort documents by score (descending)
                ranked_indices = scores.argsort()[::-1]
                
                # Select top-K documents after re-ranking
                top_docs = [docs[idx] for idx in ranked_indices[:final_results]]
                
                logger.info(f"?? Re-ranked and selected top-{final_results} chunk(s)")
                logger.debug(f"Top re-ranking scores: {sorted(scores, reverse=True)[:final_results]}")
                
                # Log the actual context being used
                for i, doc in enumerate(top_docs, 1):
                    preview = doc[:200] + '...' if len(doc) > 200 else doc
                    logger.info(f"   Context chunk {i}: {preview}")
                
                context = "\n\n".join(top_docs)
                return context
            
            return ""
            
        except Exception as e:
            logger.exception("Error retrieving KB context.")
            return ""

    async def _compress_conversation_history(self, conversation_history, max_messages: int = 6):
        """
        Compress conversation history to reduce context length while maintaining coherence.
        Keeps: most recent messages + system-critical info
        
        Args:
            conversation_history: Full conversation history
            max_messages: Maximum number of recent messages to keep (default: 6)
        
        Returns:
            Compressed conversation history
        """
        if len(conversation_history) <= max_messages:
            return conversation_history
        
        # Strategy: Keep the most recent N messages (sliding window)
        # This maintains immediate context while reducing token count
        compressed = conversation_history[-max_messages:]
        
        logger.info(f"?? Compressed conversation: {len(conversation_history)} ? {len(compressed)} messages")
        return compressed

    async def get_response(self, conversation_history):
        """Get response from LLM API, injecting Zenius KB context dynamically."""
        try:
            # First compress the conversation history to manage context length
            compressed_history = await self._compress_conversation_history(conversation_history)
            
            # Extract latest user message for retrieval
            user_query = ""
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break

            if not user_query:
                logger.warning("?? No user query found in conversation history")
                return None

            # 1?? Fetch relevant KB context from Chroma (top-10, re-ranked to top-1)
            kb_context = await self._retrieve_context(
                user_query, 
                initial_results=10, 
                final_results=1
            )

            # 2?? Prepare messages for LLM API
            messages = []
            context_injected = False
            
            for msg in compressed_history:
                content = msg["content"]
                
                # Inject KB context ONLY into the last user message (current query)
                # Check by comparing with user_query AND ensuring we haven't injected yet
                if (msg["role"] == "user" and 
                    content == user_query and 
                    kb_context and 
                    not context_injected):
                    
                    content = (
                        f"You should respond as a Voice Assistant of Zenius IT services. "
                        f"### Relevant Knowledge Base Information ###\n"
                        f"{kb_context}\n"
                        f"### End of Knowledge Base Information ###\n\n"
                        f"User Question: {content}"
                    )
                    context_injected = True
                    logger.info(f"?? Injected KB context into the latest user message")
                
                messages.append({
                    "role": msg["role"],
                    "content": content
                })

            # Log final messages being sent
            logger.info(f"?? Sending {len(messages)} messages to LLM")
            for i, msg in enumerate(messages, 1):
                preview = msg['content'][:200] + '...' if len(msg['content']) > 200 else msg['content']
                logger.info(f"   Message {i} [{msg['role'].upper()}]: {preview}")

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