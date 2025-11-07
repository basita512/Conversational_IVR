"""
Coqui TTS client for text-to-speech conversion.
"""
import os
import time
from typing import Optional
from TTS.api import TTS
import logging

logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", output_dir: str = "/tmp/tts"):
        """Initialize TTS client.

        Args:
            model_name: Name of the TTS model to use
            output_dir: Directory to save generated audio files
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tts = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    async def initialize(self):
        """Initialize the TTS model."""
        try:
            self.tts = TTS(model_name=self.model_name)
            logger.info(f"Successfully initialized TTS model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing TTS model: {e}")
            return False

    async def generate_speech(self, text: str, uuid: Optional[str] = None) -> Optional[str]:
        """Generate speech from text.

        Args:
            text: Text to convert to speech
            uuid: Optional UUID for the file name

        Returns:
            str: Path to the generated audio file or None if failed
        """
        if not self.tts:
            logger.error("TTS model not initialized")
            return None

        try:
            # Clean up old files first
            await self.cleanup_old_files()

            # Generate new file
            file_name = f"response_{uuid}.wav" if uuid else "response.wav"
            output_path = os.path.join(self.output_dir, file_name)
            
            # Generate speech with specified settings
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                sample_rate=8000,  # Required sample rate for FreeSWITCH
                save_attention=False,
                progress_bar=False
            )
            
            logger.info(f"Successfully generated speech at {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None

    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files.

        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            current_time = time.time()
            for filename in os.listdir(self.output_dir):
                if not filename.endswith('.wav'):
                    continue
                
                filepath = os.path.join(self.output_dir, filename)
                file_age = current_time - os.path.getmtime(filepath)
                
                # Delete files older than max_age_hours
                if file_age > (max_age_hours * 3600):
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audio file: {filename}")
                    except OSError as e:
                        logger.warning(f"Failed to delete old audio file {filename}: {e}")

        except Exception as e:
            logger.error(f"Error during audio file cleanup: {e}")