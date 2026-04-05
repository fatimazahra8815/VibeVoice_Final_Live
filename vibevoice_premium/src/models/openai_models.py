"""
OpenAI-compatible TTS API Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="Carter", description="Voice ID")
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speed (not yet supported)")
    stream: bool = Field(default=False, description="Enable streaming response")


class VoiceInfo(BaseModel):
    """Voice information"""
    voice_id: str
    name: str
    type: str
    gender: Optional[str] = None


class VoicesResponse(BaseModel):
    """Response for /v1/audio/voices endpoint"""
    voices: List[VoiceInfo]