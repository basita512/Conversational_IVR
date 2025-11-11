"""
Coqui TTS client for text-to-speech conversion.
"""
import os
import time
from typing import Optional
from TTS.api import TTS 
import logging
import re

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

            # Generate new file with timestamp and response counter to prevent overwriting
            import time
            timestamp = int(time.time())
            
            # Count existing responses for this UUID to get the next number
            response_count = 0
            if uuid:
                # Count existing response files for this UUID
                pattern = f"response_{uuid}_*"
                response_count = len([f for f in os.listdir(self.output_dir) 
                                   if f.startswith(f"response_{uuid}_") and f.endswith('.wav')])
            
            # Increment counter for the new response
            response_count += 1
            
            if uuid:
                file_name = f"response_{uuid}_{response_count:02d}_{timestamp}.wav"
            else:
                file_name = f"response_{response_count:02d}_{timestamp}.wav"
            output_path = os.path.join(self.output_dir, file_name)

            # Try generating speech with the original text first.
            try:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    sample_rate=8000,  # Required sample rate for FreeSWITCH
                    save_attention=False,
                    progress_bar=False
                )
                logger.info(f"Successfully generated speech at {output_path}")
                return output_path
            except Exception as primary_err:
                # Primary TTS generation failed. Log and attempt a sanitized fallback.
                logger.warning(f"Primary TTS generation failed: {primary_err}")

            # Fallback 1: sanitize text (remove non-ascii / emojis and unsupported chars)
            try:
                safe_text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove non-ascii
                safe_text = safe_text.replace('%', ' percent')
                safe_text = safe_text.strip()
                if len(safe_text) >= 3:
                    self.tts.tts_to_file(
                        text=safe_text,
                        file_path=output_path,
                        sample_rate=8000,
                        save_attention=False,
                        progress_bar=False
                    )
                    logger.info(f"Successfully generated speech at {output_path} (sanitized)")
                    return output_path
            except Exception as san_err:
                logger.warning(f"Sanitized TTS generation failed: {san_err}")

            # Fallback 2: try joining sentences into a single short paragraph and retry
            try:
                # Simple sentence splitter on periods/newlines — join into one chunk
                parts = re.split(r"[\.\n]+", text)
                joined = ' '.join([p.strip() for p in parts if len(p.strip()) > 3])
                if joined and len(joined) >= 3:
                    self.tts.tts_to_file(
                        text=joined,
                        file_path=output_path,
                        sample_rate=8000,
                        save_attention=False,
                        progress_bar=False
                    )
                    logger.info(f"Successfully generated speech at {output_path} (joined)")
                    return output_path
            except Exception as join_err:
                logger.warning(f"Joined-text TTS generation failed: {join_err}")

            # All attempts failed
            logger.error("All TTS generation attempts failed")
            return None

        except Exception as e:
            logger.exception(f"Unexpected error generating speech: {e}")
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