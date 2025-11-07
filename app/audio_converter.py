"""
Audio conversion utilities for FreeSWITCH compatibility.
"""
import soundfile as sf
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def convert_to_8khz_mono(input_path: str, output_path: str) -> Optional[str]:
    """Convert audio file to 8kHz mono WAV format for FreeSWITCH.

    Args:
        input_path: Path to input audio file
        output_path: Path to save converted audio file

    Returns:
        str: Path to converted file or None if conversion failed
    """
    try:
        # Read the audio file
        data, sample_rate = sf.read(input_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample to 8kHz if necessary
        if sample_rate != 8000:
            # Calculate number of samples for 8kHz
            n_samples = int(len(data) * 8000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data), n_samples),
                np.arange(len(data)),
                data
            )

        # Save as WAV with specific format
        sf.write(output_path, data, 8000, subtype='PCM_16')
        logger.info(f"Successfully converted audio to 8kHz mono: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None