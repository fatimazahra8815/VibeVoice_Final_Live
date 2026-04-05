"""
TTS Streaming Service
"""

import copy
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

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


class PremiumStreamingService:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.torch_device = torch.device(device)
        self.processor = None
        self.model = None
        self.voice_presets = {}
        self._voice_cache = {}

    def load(self):
        print(f"[{get_timestamp()}] Loading processor...")
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
                print(f"[{get_timestamp()}] Attempting to load model on {attempt_device.upper()}...")
                self._load_model_on_device(attempt_device)
                self.device = attempt_device  # Update actual device used
                print(f"[{get_timestamp()}] Successfully loaded model on {attempt_device.upper()}")
                return
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"[{get_timestamp()}] Failed to load on {attempt_device.upper()}: {e}")
                    if attempt_device != "cpu":
                        print(f"[{get_timestamp()}] Falling back to next device...")
                        continue
                raise e
            except Exception as e:
                print(f"[{get_timestamp()}] Unexpected error loading on {attempt_device.upper()}: {e}")
                if attempt_device != "cpu":
                    continue
                raise e
        
        raise RuntimeError("Failed to load model on any available device")

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

        print(f"\n" + "="*50)
        print(f" DEVICE: {device.upper()} ".center(50, "="))
        print("="*50 + "\n")
        
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
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )

        # Ensure speech factors are valid (not NaN)
        if torch.isnan(self.model.model.speech_scaling_factor).any():
            print(f"[{get_timestamp()}] Warning: speech_scaling_factor is NaN, initializing to default 1.15")
            self.model.model.speech_scaling_factor.fill_(1.15)
        if torch.isnan(self.model.model.speech_bias_factor).any():
            print(f"[{get_timestamp()}] Warning: speech_bias_factor is NaN, initializing to default 0.0")
            self.model.model.speech_bias_factor.fill_(0.0)

    def unload(self):
        if self.model is not None:
             print(f"[{get_timestamp()}] Unloading TTS model...")
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
        print(f"[{get_timestamp()}] TTS model unloaded.")

    def refresh_voices(self):
        self.voice_presets = {}
        # Load built-in presets
        if VOICES_DIR.exists():
            for pt_path in VOICES_DIR.rglob("*.pt"):
                self.voice_presets[pt_path.stem] = pt_path

        print(f"[{get_timestamp()}] Loaded {len(self.voice_presets)} voices total.")

    def get_voice_prompt(self, voice_key: str):
        if voice_key not in self.voice_presets:
            if self.voice_presets:
                voice_key = next(iter(self.voice_presets))
            else:
                return None, None

        if voice_key not in self._voice_cache:
            path = self.voice_presets[voice_key]
            data = torch.load(path, map_location=self.torch_device, weights_only=False)

            # Wrap dictionaries in a class that allows attribute access if needed by library
            from transformers.modeling_outputs import BaseModelOutputWithPast

            def wrap_output(d):
                if isinstance(d, dict) and "past_key_values" in d:
                    return BaseModelOutputWithPast(**d)
                return d

            if isinstance(data, dict):
                for k in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
                    if k in data:
                        data[k] = wrap_output(data[k])

            self._voice_cache[voice_key] = data

        return voice_key, self._voice_cache[voice_key]

    def ensure_loaded(self):
         if self.model is None:
             self.load()

    def stream(self, text: str, voice_key: str, cfg_scale=1.5, steps=5, log_callback=None, stop_event=None):
        # MANAGED SWAP: Ensure Podcast is unloaded and TTS is loaded
        if hasattr(self, '_app') and self._app.state.podcast_service:
             self._app.state.podcast_service.unload_model()
        self.ensure_loaded()

        if not text.strip(): return

        selected_key, prompt = self.get_voice_prompt(voice_key)
        self.model.set_ddpm_inference_steps(num_steps=steps)

        # Debug scaling factors
        s_val = self.model.model.speech_scaling_factor.item()
        b_val = self.model.model.speech_bias_factor.item()
        print(f"[{get_timestamp()}] Synthesis start: voice={voice_key}, scale={s_val:.4f}, bias={b_val:.4f}")

        # Prepare inputs
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip().replace("'", "'"),
            cached_prompt=prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = {k: v.to(self.torch_device) if hasattr(v, "to") else v for k, v in processed.items()}

        audio_streamer = AudioStreamer(batch_size=1)
        errors = []
        stop_signal = stop_event or threading.Event()

        def run_gen():
            try:
                self.model.generate(
                    **inputs,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_signal.is_set,
                    all_prefilled_outputs=copy.deepcopy(prompt)
                )
            except Exception as e:
                print(f"[{get_timestamp()}] Generation error: {e}")
                traceback.print_exc()
                errors.append(e)
                audio_streamer.end()

        thread = threading.Thread(target=run_gen, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for chunk in stream:
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                yield chunk.reshape(-1)
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors and log_callback:
                log_callback("error", str(errors[0]))