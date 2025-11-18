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
    def __init__(self, model_name: str = "tts_models/en/ljspeech/glow-tts", output_dir: str = "/tmp/tts"):
        """Initialize TTS client.

        Args:
            model_name: Name of the TTS model to use
            output_dir: Directory to save generated audio files
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tts = None
        self.initialized = False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    async def initialize(self):
        """Initialize the TTS model with optimized settings."""
        try:
            # Initialize the TTS model first
            self.tts = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=False  # Set to True if you have a CUDA-capable GPU
            )
            
            # Warm up the model with a short phrase
            warmup_path = os.path.join(self.output_dir, "warmup.wav")
            self.tts.tts_to_file(
                text="Hello, I'm initializing.",
                file_path=warmup_path,
                speaker_wav=None
            )
            
            # Clean up the warmup file
            try:
                if os.path.exists(warmup_path):
                    os.remove(warmup_path)
            except Exception as e:
                logger.warning(f"Could not remove warmup file: {e}")
                
            self.initialized = True
            logger.info(f"Successfully initialized TTS model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing TTS model: {e}", exc_info=True)
            self.initialized = False
            return False

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing special characters and normalizing whitespace.
        
        Args:
            text: The text to sanitize
            
        Returns:
            str: Sanitized text
        """
        import re
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text before TTS by removing intent tags and non-ASCII characters.
        
        Args:
            text: The text to clean
            
        Returns:
            str: Cleaned text with intent tags and non-ASCII characters removed
        """
        import re
        
        # First remove intent tags like <intent>sales</intent>
        cleaned = re.sub(r'<intent>.*?</intent>', '', text)
        
        # Then remove non-ASCII characters
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
        
        # Clean up any extra whitespace that might have been left
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    async def generate_speech(self, text: str, uuid: Optional[str] = None) -> Optional[str]:
        """Generate speech from text.

        Args:
            text: Text to convert to speech
            uuid: Optional UUID for the file name

        Returns:
            str: Path to the generated audio file or None if failed
        """
        if not self.initialized:
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
                # Match files with pattern: response_XX_uuid_*.wav
                response_count = len([f for f in os.listdir(self.output_dir) 
                                   if f.startswith(f"response_") and f"_{uuid}_" in f and f.endswith('.wav')])
            
            # Increment counter for the new response
            response_count += 1
            
            if uuid:
                file_name = f"response_{response_count:02d}_{uuid}_{timestamp}.wav"
            else:
                file_name = f"response_{response_count:02d}_{timestamp}.wav"
            output_path = os.path.join(self.output_dir, file_name)

            # Clean the text by removing intent tags before TTS
            clean_text = self._clean_text_for_tts(text)
            logger.debug(f"Original text: {text}")
            logger.debug(f"Cleaned text for TTS: {clean_text}")

            # Try generating speech with the cleaned text
            try:
                self.tts.tts_to_file(
                    text=clean_text,
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
                safe_text = self._sanitize_text(clean_text)
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
                parts = re.split(r"[\.\n]+", clean_text)
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

    async def cleanup_old_files(self, max_age_hours: int = 24, max_files: int = 100):
        """Clean up old audio files and enforce maximum file count.

        Args:
            max_age_hours: Maximum age of files to keep in hours
            max_files: Maximum number of files to keep in the directory
        """
        try:
            current_time = time.time()
            files = []
            
            # First, collect all wav files with their modification times
            for filename in os.listdir(self.output_dir):
                if not filename.endswith('.wav'):
                    continue
                filepath = os.path.join(self.output_dir, filename)
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime, current_time - mtime))
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Remove files older than max_age_hours
            for filepath, _, file_age in files:
                if file_age > (max_age_hours * 3600):
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audio file: {os.path.basename(filepath)}")
                    except OSError as e:
                        logger.warning(f"Failed to delete old audio file {filepath}: {e}")
            
            # If still too many files, remove the oldest ones
            files = [(f, m) for f, m, _ in files if os.path.exists(f)]  # Refresh file list
            if len(files) > max_files:
                for filepath, _ in files[:len(files) - max_files]:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up excess audio file: {os.path.basename(filepath)}")
                    except OSError as e:
                        logger.warning(f"Failed to delete excess audio file {filepath}: {e}")

        except Exception as e:
            logger.error(f"Error during audio file cleanup: {e}", exc_info=True)