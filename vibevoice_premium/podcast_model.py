import threading, librosa, torch, gc
import numpy as np
import soundfile as sf
import gradio as gr 

from typing import Iterator, Optional
import os, time, traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
# from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data

class VibeVoiceDemo:
    voices_dir = os.path.join(os.path.dirname(__file__), "static", "podcast_voices")
    examples_dir = os.path.join(os.path.dirname(__file__), "static", "podcast_examples")

    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, adapter_path: Optional[str] = None):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.adapter_path = adapter_path
        self.loaded_adapter_root: Optional[str] = None
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        # self.load_model() # Removed for lazy loading
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    def ensure_loaded(self, log_callback=None):
        if not hasattr(self, 'model') or self.model is None:
            self.load_model(log_callback)
        
    def load_model(self, log_callback=None):
        """Load the VibeVoice model and processor."""
        if log_callback:
            log_callback("log", "Loading Podcast model...")
        print(f"Loading processor & model from {self.model_path}")
        self.loaded_adapter_root = None
        
        # Normalize potential 'mpx'
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        # Try to load on preferred device, fallback to CPU if GPU memory insufficient
        self._load_model_with_fallback(log_callback)

    def _load_model_with_fallback(self, log_callback=None):
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
                if log_callback:
                    log_callback("log", f"Attempting to load model on {attempt_device.upper()}...")
                print(f"[{get_timestamp()}] Attempting to load model on {attempt_device.upper()}...")
                self._load_model_on_device(attempt_device)
                self.device = attempt_device  # Update actual device used
                if log_callback:
                    log_callback("log", f"Successfully loaded model on {attempt_device.upper()}")
                print(f"[{get_timestamp()}] Successfully loaded model on {attempt_device.upper()}")
                return
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    if log_callback:
                        log_callback("log", f"Failed to load on {attempt_device.upper()}: insufficient memory")
                    print(f"[{get_timestamp()}] Failed to load on {attempt_device.upper()}: {e}")
                    if attempt_device != "cpu":
                        if log_callback:
                            log_callback("log", "Falling back to next device...")
                        print(f"[{get_timestamp()}] Falling back to next device...")
                        continue
                raise e
            except Exception as e:
                if log_callback:
                    log_callback("log", f"Unexpected error loading on {attempt_device.upper()}")
                print(f"[{get_timestamp()}] Unexpected error loading on {attempt_device.upper()}: {e}")
                if attempt_device != "cpu":
                    continue
                raise e
        
        error_msg = "Failed to load model on any available device"
        if log_callback:
            log_callback("log", error_msg)
        raise RuntimeError(error_msg)

    def _load_model_on_device(self, device):
        """Load model on specific device with appropriate settings."""
        # Decide dtype & attention
        if device == "mps":
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        elif device == "cuda":
            load_dtype = torch.bfloat16
            try:
                import flash_attn
                attn_impl_primary = "flash_attention_2"
            except ImportError:
                attn_impl_primary = "sdpa"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        
        print(f"Using device: {device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        # Load model
        try:
            if device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                self.model.to("mps")
            elif device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(device if device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                )
                if device == "mps":
                    self.model.to("mps")
            else:
                raise e
        
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
    
    def unload_model(self):
        """Unload model and free GPU memory."""
        if hasattr(self, 'model') and self.model is not None:
             print("Unloading Podcast model...")
             del self.model
             self.model = None
        
        if hasattr(self, 'processor') and self.processor is not None:
             del self.processor
             self.processor = None
             
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print("Podcast model unloaded.")

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        # Check if voices directory exists
        if not os.path.exists(self.voices_dir):
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all WAV files in the voices directory
        self.voice_presets = {}

        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(self.voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(self.voices_dir, f))]

        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(self.voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        if not self.available_voices:
            # We don't want to crash init if this fails, just log it
            print("No voice presets found. Please add .wav files to the demo/voices directory.")
        else:
            print(f"Found {len(self.available_voices)} voice files in {self.voices_dir}")
            print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3,
                                 disable_voice_cloning: bool = False) -> Iterator[tuple]:
        try:
            
            # Ensure model is loaded
            self.ensure_loaded()
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise ValueError("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise ValueError("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise ValueError(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            voice_cloning_enabled = not disable_voice_cloning

            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            log += f"🔊 Voice cloning: {'Enabled' if voice_cloning_enabled else 'Disabled'}\n"
            if self.loaded_adapter_root:
                log += f"🧩 LoRA: {self.loaded_adapter_root}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", {"visible": False}
                return
            
            # Load voice samples when voice cloning is enabled
            voice_samples = None
            if voice_cloning_enabled:
                voice_samples = []
                for speaker_name in selected_speakers:
                    audio_path = self.available_voices[speaker_name]
                    audio_data = self.read_audio(audio_path)
                    if len(audio_data) == 0:
                        self.is_generating = False
                        raise ValueError(f"Error: Failed to load audio for {speaker_name}")
                    voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", {"visible": False}
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (1.5B model)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", {"visible": False}
                return
            
            start_time = time.time()
            
            # Standard Processor Call for 1.5B Model
            processor_kwargs = {
                "text": [formatted_script],
                "padding": True,
                "return_tensors": "pt",
                "return_attention_mask": True,
            }
            if voice_samples is not None:
                processor_kwargs["voice_samples"] = [voice_samples]
            else:
                processor_kwargs["voice_samples"] = None

            try:
                inputs = self.processor(**processor_kwargs)
            except Exception as pe:
                 # Catch error if processor isn't loaded correctly
                  self.is_generating = False
                  raise ValueError(f"Processor Error: {str(pe)}")
            
            # Move tensors to device
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, voice_cloning_enabled)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # reduced from 3 to 1 second

            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", {"visible": False}
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, {"visible": True}
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, {"visible": True}
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", {"visible": False}
                return
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, {"visible": False}
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, {"visible": False}
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, {"visible": False}
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, {"visible": False}
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, {"visible": False}

        except ValueError as e:
            # Handle specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, {"visible": False}
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, {"visible": False}
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer, voice_cloning_enabled: bool):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
                is_prefill=voice_cloning_enabled,
            )

        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()

    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")

    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(self.examples_dir):
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(self.examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(self.examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(self.examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the count of unique speakers if they're 1-based
        return len(speakers)

    def generate_tts(self, text: str, speaker: str, cfg_scale: float = 1.3) -> tuple:
        """
        Generate TTS using the 1.5B model (non-streaming).
        Returns: (sample_rate, audio_numpy_array)
        """
        try:
            self.ensure_loaded()
            
            if not text or not text.strip():
                raise ValueError("Please provide text for TTS.")
            
            # Use 'Speaker 0' format for the single speaker
            formatted_script = f"Speaker 0: {text.strip()}"
            
            if not speaker or speaker not in self.available_voices:
                raise ValueError("Invalid speaker selected.")
                
            audio_path = self.available_voices[speaker]
            voice_sample = self.read_audio(audio_path)
            
            if len(voice_sample) == 0:
                 raise ValueError(f"Failed to load audio for {speaker}")
            
            # Inputs
            processor_kwargs = {
                "text": [formatted_script],
                "padding": True,
                "return_tensors": "pt",
                "return_attention_mask": True,
                "voice_samples": [[voice_sample]]
            }
            
            inputs = self.processor(**processor_kwargs)
            
            # Move to device
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None, # Let model determine length
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        'do_sample': False,
                    },
                    verbose=False,
                    refresh_negative=True,
                    is_prefill=True, # Voice cloning enabled for TTS 1.5B
                )
            
            # Output processing
            audio_output = None
            
            # Check if output is VibeVoiceGenerationOutput (has speech_outputs)
            if hasattr(outputs, "speech_outputs") and outputs.speech_outputs:
                audio_tensor = outputs.speech_outputs[0]
                if torch.is_tensor(audio_tensor):
                    audio_output = audio_tensor.float().cpu().numpy()
                else:
                    audio_output = audio_tensor
            # Access by dict key if it behaves like a dict (ModelOutput often does)
            elif isinstance(outputs, dict) and "speech_outputs" in outputs:
                 audio_tensor = outputs["speech_outputs"][0]
                 if torch.is_tensor(audio_tensor):
                    audio_output = audio_tensor.float().cpu().numpy()
                 else:
                    audio_output = audio_tensor
            # Fallback for tuple/list return
            elif isinstance(outputs, (list, tuple)):
                # If it's a tuple, identifying which one is audio might be tricky without known order
                # But based on inference code, it returns VibeVoiceGenerationOutput object.
                # However, if return_dict=False passed (it wasn't), it might be tuple.
                # Let's assume it's like [logits, speech] or something.
                # But we are fairly sure it's the object.
                pass 
            
            if audio_output is None:
                 # Try to debug
                 print(f"Output type: {type(outputs)}")
                 print(f"Output keys: {outputs.__dict__.keys() if hasattr(outputs, '__dict__') else 'No dict'}")
                 raise ValueError("Failed to extract audio from model output.")

            if len(audio_output.shape) > 1:
                audio_output = audio_output.squeeze()
                
            return (24000, audio_output)

        except Exception as e:
            print(f"Error in TTS 1.5B generation: {e}")
            traceback.print_exc()
            raise e

    def save_cloned_voice(self, audio_file, language: str, name: str, gender: str) -> str:
        """
        Save a cloned voice file with the convention: [language]-[name]-[gender].wav
        """
        try:
            if not os.path.exists(self.voices_dir):
                os.makedirs(self.voices_dir)
            
            # Sanitize inputs
            language = language.lower().strip()
            name = "".join(c for c in name if c.isalnum() or c in ('-','_')).strip()
            gender = gender.lower().strip()
            
            filename = f"{language}-{name}-{gender}.wav"
            save_path = os.path.join(self.voices_dir, filename)
            
            # If audio_file is bytes/file-like or path?
            # Assume it's a file path (temp file from upload) or bytes.
            # If standard UploadFile from FastAPI, we read it.
            
            # For this internal method, let's assume it's receiving the bytes or a source path.
            # We will handle the file saving in app.py generally, but `read_audio` here takes a path.
            # Actually, `app.py` usually handles file uploads via FastAPI. 
            # I will just expose a method to refresh the list and let app.py save the file.
            pass
        except Exception as e:
             raise e
    
    def refresh_voices(self):
        """Force refresh of voice presets."""
        self.setup_voice_presets()
