import asyncio
import logging
import os
import time
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Import and include routers after app is created
from app.freeswitch_client import router as freeswitch_router
app.include_router(freeswitch_router)

# For testing allow everything (NOT recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # change to your allowed origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Conversational_IVR instance (created and initialized at startup)
agent = None

from app.config import settings
from app.llm_client import LLMClient
from app.tts_client import TTSClient
from app.conversation import Conversation

# Configure logging (level can be overridden with LOG_LEVEL env var)
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    log_level = getattr(logging, log_level_str)
except Exception:
    log_level = logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


async def warm_up_llm():
    """Preload Ollama model to prevent first-request cold start."""
    try:
        llm = LLMClient()
        payload = {
            "model": "zenius-llm",
            "messages": [{"role": "user", "content": "Warmup request - respond with OK"}],
            "stream": False
        }
        await llm.client.post(llm.api_url, json=payload)
        logger.info("🔥 LLM model preloaded successfully (no cold start expected).")
        await llm.close()
    except Exception as e:
        logger.warning(f"Warm-up failed: {e}")

# # Call this in app startup event
# @app.on_event("startup")
# async def on_startup():
#     await warm_up_llm()


class Conversational_IVR:
    def __init__(self):
        """Initialize the support agent components."""
        # Initialize components
        self.llm_client = LLMClient()
        self.tts_client = TTSClient(output_dir=settings.tts_output_dir)
        self.conversation = Conversation()

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize TTS client
            if not await self.tts_client.initialize():
                return False
            
            logger.info("Successfully initialized all components")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False

    async def handle_transcription(self, event: Dict[str, Any]):
        """Handle transcription events from FreeSWITCH.

        Args:
            event: Event data containing call_uuid and transcription
        """
        try:
            call_uuid = event['call_uuid']
            transcription = event['transcription']

            # Add user message to conversation history
            self.conversation.add_message(call_uuid, 'user', transcription)

            # Get conversation history
            history = self.conversation.get_history(call_uuid)
            logger.debug(f"Conversation history for {call_uuid} (count={len(history)}): {history}")

            # Get LLM response
            response = await self.llm_client.get_response(history)

            if response:
                # Add assistant response to conversation history
                self.conversation.add_message(call_uuid, 'assistant', response)

                # Generate speech from response
                audio_path = await self.tts_client.generate_speech(response, call_uuid)

                if audio_path:
                    logger.info(f"Generated audio at {audio_path}")
                else:
                    logger.error("Failed to generate speech from response")
            else:
                logger.error("Failed to get LLM response")

        except Exception as e:
            logger.error(f"Error handling transcription: {e}")

    async def run(self):
        """Run the support agent."""
        try:
            if await self.initialize():
                logger.info("Support agent is running in API mode")
                # Keep the application running
                while True:
                    await asyncio.sleep(3600)  # Sleep for 1 hour
            else:
                logger.error("Failed to initialize support agent")
        except Exception as e:
            logger.error(f"Error running support agent: {e}")

def main():
    """Main entry point."""
    agent = Conversational_IVR()
    asyncio.run(agent.run())


# Create the shared agent instance and wire it into FastAPI lifecycle events
agent = Conversational_IVR()


@app.on_event("startup")
async def startup_event():
    await warm_up_llm()
    """Initialize shared agent on application startup."""
    logger.info("Starting application startup: initializing Conversational_IVR...")
    ok = await agent.initialize()
    if not ok:
        logger.error("Conversational_IVR failed to initialize at startup")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Conversational_IVR and cleaning up resources...")
    try:
        if hasattr(agent, 'llm_client') and agent.llm_client:
            await agent.llm_client.close()
            logger.info("LLM client closed successfully")
    except Exception as e:
        logger.error(f"Error during LLM client shutdown: {str(e)}")
    
    try:
        if hasattr(agent, 'tts_client') and agent.tts_client:
            # Add any TTS client cleanup if needed
            logger.info("TTS client cleaned up")
    except Exception as e:
        logger.error(f"Error during TTS client cleanup: {str(e)}")
    
    logger.info("Shutdown completed")

if __name__ == "__main__":
    main()
