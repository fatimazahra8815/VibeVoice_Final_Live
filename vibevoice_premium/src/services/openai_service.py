"""
OpenAI-compatible TTS Service
"""

import copy
import threading
import traceback
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from transformers import logging as hf_logging

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# Suppress "Some weights were not initialized" warnings
hf_logging.set_verbosity_error()

# Paths
VOICES_DIR = Path(__file__).parent.parent.parent / "voices" / "streaming_model"


def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class OpenAITTSService:
    """Service for OpenAI-compatible TTS using VibeVoice 0.5B"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor = None
        self.model = None
        self.voice_presets = {}
        self._voice_cache = {}
        self.torch_device = torch.device(device)

    def load(self):
        print(f"[{get_timestamp()}] Loading OpenAI TTS processor...")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Try to load on preferred device, fallback to CPU if GPU memory insufficient
        self._load_model_with_fallback()

        # Load voices
        self.refresh_voices()

    def _load_model_with_fallback(self):
        """Load model with automatic GPU->CPU fallback on memory issues."""
        devices_to_try = []
        
        # Prefer the requested device, but try GPU first if available
        if self.device == "cuda" and torch.cuda.is_available():
            devices_to_try = ["cuda", "cpu"]
        elif self.device == "mps" and torch.backends.mps.is_available():
            devices_to_try = ["mps", "cpu"]
        else:
            devices_to_try = ["cpu"]
        
        for attempt_device in devices_to_try:
            try:
                print(f"[{get_timestamp()}] Attempting to load OpenAI TTS model on {attempt_device.upper()}...")
                self._load_model_on_device(attempt_device)
                self.device = attempt_device  # Update actual device used
                self.torch_device = torch.device(attempt_device)
                print(f"[{get_timestamp()}] Successfully loaded OpenAI TTS model on {attempt_device.upper()}")
                return
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"[{get_timestamp()}] Failed to load OpenAI TTS on {attempt_device.upper()}: {e}")
                    if attempt_device != "cpu":
                        print(f"[{get_timestamp()}] Falling back to next device...")
                        continue
                raise e
            except Exception as e:
                print(f"[{get_timestamp()}] Unexpected error loading OpenAI TTS on {attempt_device.upper()}: {e}")
                if attempt_device != "cpu":
                    continue
                raise e
        
        raise RuntimeError("Failed to load OpenAI TTS model on any available device")

    def _load_model_on_device(self, device):
        """Load model on specific device with appropriate settings."""
        # Determine dtype and attention
        if device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device if device == "cuda" else None,
                attn_implementation=attn_impl,
            )
            if device == "mps":
                self.model.to("mps")
        except Exception:
            print(f"[{get_timestamp()}] Flash Attention failed, falling back to SDPA")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device if device != "mps" else None,
                attn_implementation="sdpa",
            )
            if device == "mps":
                self.model.to("mps")

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)

    def unload(self):
        if self.model is not None:
             print(f"[{get_timestamp()}] Unloading OpenAI TTS model...")
             del self.model
             self.model = None
        if self.processor is not None:
             del self.processor
             self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()

    def refresh_voices(self):
        self.voice_presets = {}
        # Load built-in presets
        if VOICES_DIR.exists():
            for pt_path in VOICES_DIR.rglob("*.pt"):
                self.voice_presets[pt_path.stem] = pt_path

        print(f"[{get_timestamp()}] Loaded {len(self.voice_presets)} voices for OpenAI TTS.")

    def get_available_voices(self):
        from ..models.openai_models import VoiceInfo
        voices = []
        # Add OpenAI-compatible voices (only if the corresponding VibeVoice voice exists)
        openai_voices = {
            "alloy": "en-Carter_man",
            "echo": "en-Davis_man",
            "fable": "en-Emma_woman",
            "onyx": "en-Frank_man",
            "nova": "en-Grace_woman",
            "shimmer": "en-Mike_man",
        }
        for openai_name, vibevoice_name in openai_voices.items():
            if vibevoice_name in self.voice_presets:
                voices.append(VoiceInfo(
                    voice_id=openai_name,
                    name=openai_name,
                    type="openai-compatible",
                    gender="man" if "man" in vibevoice_name else "woman" if "woman" in vibevoice_name else None
                ))

        # Add native VibeVoice voices (without language prefix for simplicity)
        for name in self.voice_presets.keys():
            if "-" in name and name not in openai_voices.values():
                # Extract just the voice name without language prefix
                voice_name = name.split("-")[1] if "-" in name else name
                voices.append(VoiceInfo(
                    voice_id=name,  # Keep full name as ID for native voices
                    name=voice_name,
                    type="vibevoice-native",
                    gender="man" if "man" in name else "woman" if "woman" in name else None
                ))
        return voices

    def _resolve_voice(self, voice: str):
        openai_voices = {
            "alloy": "en-Carter_man",
            "echo": "en-Davis_man",
            "fable": "en-Emma_woman",
            "onyx": "en-Frank_man",
            "nova": "en-Grace_woman",
            "shimmer": "en-Mike_man",
        }
        if voice.lower() in openai_voices:
            voice = openai_voices[voice.lower()]
        if voice not in self.voice_presets:
            # Try to find a voice that starts with the requested name
            for preset_name in self.voice_presets.keys():
                if voice in preset_name:
                    return preset_name
            # Default fallback
            return list(self.voice_presets.keys())[0] if self.voice_presets else voice
        return voice

    def _get_voice_prompt(self, voice: str):
        if voice not in self._voice_cache:
            path = self.voice_presets[voice]
            data = torch.load(path, map_location=self.torch_device, weights_only=False)
            if isinstance(data, dict) and "past_key_values" in data:
                from transformers.modeling_outputs import BaseModelOutputWithPast
                data = BaseModelOutputWithPast(**data)
            self._voice_cache[voice] = data
        return self._voice_cache[voice]

    def generate_speech(self, text: str, voice: str, cfg_scale: float = 1.25):
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)

        text = text.strip().replace("'", "'")

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = {k: v.to(self.torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        print(f"[{get_timestamp()}] Generating OpenAI TTS for {len(text)} chars with voice '{voice}'")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
        )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.reshape(-1)
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak
            return audio
        else:
            raise RuntimeError("No audio output generated")

    def generate_speech_streaming(self, text: str, voice: str, cfg_scale: float = 1.25):
        """Generate speech streaming from text"""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)

        text = text.strip().replace("'", "'")

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = {k: v.to(self.torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        print(f"[{get_timestamp()}] Streaming OpenAI TTS for {len(text)} chars with voice '{voice}'")

        audio_streamer = AudioStreamer(batch_size=1)

        def run_gen():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    audio_streamer=audio_streamer,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
            except Exception as e:
                print(f"[{get_timestamp()}] Streaming generation error: {e}")
                traceback.print_exc()
                audio_streamer.end()

        thread = threading.Thread(target=run_gen, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for chunk in stream:
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                chunk = chunk.reshape(-1)
                # Convert to PCM16
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                yield pcm
        finally:
            audio_streamer.end()
            thread.join()