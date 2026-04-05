"""
Audio processing utilities
"""

import numpy as np
import scipy.io.wavfile as wavfile


def convert_audio(audio: np.ndarray, format: str, sample_rate: int = 24000) -> bytes:
    """Convert audio to specified format
    
    Currently supports:
    - pcm: Raw PCM data (16-bit signed integers)
    - wav: WAV format
    """
    format = format.lower()
    if format == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()
    if format == "wav":
        import io
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
        return buffer.getvalue()
    # Fallback to WAV for unsupported formats
    import io
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
    return buffer.getvalue()


def get_content_type(format: str) -> str:
    """Get MIME content type for audio format"""
    types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }
    return types.get(format.lower(), "application/octet-stream")