"""
Main application entry point.
Handles ESL connection and event processing.
"""
import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()

class TranscriptionEvent(BaseModel):
    call_uuid: str
    transcription: str

@app.post("/test/transcription")
async def test_transcription(event: TranscriptionEvent):
    # Simulate the ESL event flow
    # Use the same logic as handle_transcription
    agent = SupportAgent()
    await agent.initialize()
    await agent.handle_transcription(event.dict())
    # Find the latest audio file for this UUID
    import os
    audio_dir = agent.tts_client.output_dir
    audio_file = os.path.join(audio_dir, f"response_{event.call_uuid}.wav")
    if os.path.exists(audio_file):
        return FileResponse(audio_file, media_type="audio/wav")
    return {"status": "completed"}

from app.config import settings
from app.esl_handler import ESLHandler
from app.llm_client import LLMClient
from app.tts_client import TTSClient
from app.conversation import Conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportAgent:
    def __init__(self):
        """Initialize the support agent components."""
        # Initialize components
        self.esl_handler = ESLHandler(
            host=settings.freeswitch_host,
            port=settings.freeswitch_port,
            password=settings.freeswitch_password
        )
        self.llm_client = LLMClient()
        self.tts_client = TTSClient(output_dir=settings.tts_output_dir)
        self.conversation = Conversation()

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Connect to FreeSWITCH ESL
            if not await self.esl_handler.connect():
                return False

            # Initialize TTS client
            if not await self.tts_client.initialize():
                return False

            # Register transcription handler
            self.esl_handler.register_handler('transcription', self.handle_transcription)
            
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

            # Get LLM response
            response = await self.llm_client.get_response(history)

            if response:
                # Add assistant response to conversation history
                self.conversation.add_message(call_uuid, 'assistant', response)

                # Generate speech from response
                audio_path = await self.tts_client.generate_speech(response, call_uuid)

                if audio_path:
                    # Send playback command to FreeSWITCH
                    await self.esl_handler.send_playback_command(call_uuid, audio_path)
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
                logger.info("Starting ESL event listener")
                await self.esl_handler.start_listening()
            else:
                logger.error("Failed to initialize support agent")
        except Exception as e:
            logger.error(f"Error running support agent: {e}")
        finally:
            self.esl_handler.disconnect()

def main():
    """Main entry point."""
    agent = SupportAgent()
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
