import datetime
import asyncio
import json
import os
import threading
import traceback
import gc
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import math
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState
from transformers import logging as hf_logging
from .podcast_model import VibeVoiceDemo

# Define base path
BASE = Path(__file__).parent

# Local imports
from .src.services.tts_service import PremiumStreamingService
from .src.services.openai_service import OpenAITTSService
from .src.models.openai_models import TTSRequest, VoicesResponse
from .src.utils.audio_utils import convert_audio, get_content_type

# Suppress "Some weights were not initialized" warnings
hf_logging.set_verbosity_error()

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve API demo
@app.get("/api-demo")
async def api_demo():
    """Serve the API demo HTML page"""
    demo_path = BASE.parent / "api_demo.html"
    if demo_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(demo_path, media_type="text/html")
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="API demo not found")

# Mount Podcast App
@app.on_event("startup")
async def startup():
    # Prefer GPU if available, but services will handle fallback to CPU if needed
    device = os.environ.get("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cuda" and torch.cuda.is_available():
        print(f"[{get_timestamp()}] Preferring CUDA (GPU) - will fallback to CPU if memory insufficient")
    elif device == "mps" and torch.backends.mps.is_available():
        print(f"[{get_timestamp()}] Using MPS (Apple Silicon)")
    else:
        print(f"[{get_timestamp()}] Using CPU")
    
    model_path = os.environ.get("MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B")
    podcast_model_path = os.environ.get("PODCAST_MODEL_PATH", "microsoft/VibeVoice-1.5B")

    print(f"\n[{get_timestamp()}] INITIALIZING VIBEVOICE PREMIUM SERVICE")
    print(f"[{get_timestamp()}] Models will be loaded on first use (Lazy Loading).")

    # Initialize TTS Service (Lazy)
    service = PremiumStreamingService(model_path, device)
    app.state.service = service
    app.state.lock = asyncio.Lock()
    
    # Initialize Podcast Service (Lazy)
    try:
        podcast_service = VibeVoiceDemo(
            model_path=podcast_model_path,
            device=device,
            inference_steps=5
        )
        app.state.podcast_service = podcast_service
        app.state.podcast_lock = asyncio.Lock()
        print(f"[{get_timestamp()}] Podcast Service initialized (Ready to load).")
    except Exception as e:
        print(f"Failed to initialize podcast service: {e}")
        traceback.print_exc()
        app.state.podcast_service = None

    # Initialize OpenAI TTS Service (Lazy)
    openai_service = OpenAITTSService(model_path, device)
    app.state.openai_service = openai_service
    app.state.openai_lock = asyncio.Lock()
    print(f"[{get_timestamp()}] OpenAI TTS Service initialized (Ready to load).")

    print(f"[{get_timestamp()}] SERVICE READY - WAITING FOR CONNECTIONS\n")

    # Open browser automatically
    webbrowser.open("http://localhost:7860")

@app.get("/")
async def get_index():
    return FileResponse(BASE / "templates" / "index.html")

@app.get("/config")
async def get_config():
    service = app.state.service
    service.refresh_voices()
    return {
        "voices": sorted(list(service.voice_presets.keys())),
        "default_voice": next(iter(service.voice_presets.keys()), None)
    }

@app.get("/stats")
async def get_system_stats():
    """Get system RAM and VRAM usage statistics."""
    import psutil
    
    # Get RAM stats
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    
    # Get VRAM stats if CUDA is available
    vram_used_gb = 0
    vram_total_gb = 0
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            vram_free, vram_total = torch.cuda.mem_get_info(device)
            vram_used_gb = (vram_total - vram_free) / (1024**3)
            vram_total_gb = vram_total / (1024**3)
        except Exception:
            pass
    
    return {
        "ram": {
            "used": round(ram_used_gb, 1),
            "total": round(ram_total_gb, 1),
            "percentage": ram.percent
        },
        "vram": {
            "used": round(vram_used_gb, 1),
            "total": round(vram_total_gb, 1),
            "percentage": round((vram_used_gb / vram_total_gb * 100) if vram_total_gb > 0 else 0, 1)
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    service = app.state.service
    lock = app.state.lock

    if lock.locked():
        await websocket.send_json({"type": "error", "message": "Backend busy"})
        await websocket.close()
        return

    async with lock:
        try:
            # First message should be configuration
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice = data.get("voice", "")
            cfg = float(data.get("cfg", 1.5))
            steps = int(data.get("steps", 5))

            stop_event = threading.Event()
            
            # Simple iterator bridge
            def gen():
                for chunk in service.stream(text, voice, cfg, steps, stop_event=stop_event):
                    # Check for NaN to prevent cast error
                    if np.isnan(chunk).any():
                        print(f"[{get_timestamp()}] Warning: Generated chunk contains NaN.")
                        chunk = np.zeros_like(chunk)
                        
                    # Convert to PCM16
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    yield pcm

            # Iterate in thread to keep websocket responsive
            loop = asyncio.get_event_loop()
            iterator = gen()

            while True:
                try:
                    chunk_bytes = await loop.run_in_executor(None, next, iterator, None)
                    if chunk_bytes is None:
                        break
                    await websocket.send_bytes(chunk_bytes)
                except StopIteration:
                    break
            
            await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except: pass
@app.get("/podcast/config")
async def get_podcast_config():
    service = app.state.podcast_service
    if not service:
        return {"error": "Podcast service not available"}
    
    # Ensure available_voices is populated (might need setup_voice_presets if not done in init)
    if not service.available_voices:
        service.setup_voice_presets()

    return {
        "voices": sorted(list(service.available_voices.keys())),
        "default_voices": list(service.available_voices.keys())[:4]
    }

@app.websocket("/podcast/ws")
async def podcast_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    service = app.state.podcast_service
    lock = app.state.podcast_lock
    
    if not service:
        await websocket.send_json({"type": "error", "message": "Service unavailable"})
        await websocket.close()
        return

    if lock.locked():
        await websocket.send_json({"type": "error", "message": "Backend busy"})
        await websocket.close()
        return

    async with lock:
        try:
            data = await websocket.receive_json()
            if data.get("type") == "stop":
                 service.stop_audio_generation()
                 return

            script = data.get("script", "")
            num_speakers = int(data.get("num_speakers", 2))
            speakers = data.get("speakers", []) # List of speaker names
            cfg_scale = float(data.get("cfg", 1.5))
            disable_cloning = data.get("disable_cloning", False)

            # Pad speakers list if needed
            while len(speakers) < 4:
                speakers.append(None)

            # Define a generator that runs the sync generator in a thread
            def run_generation():
                # MANAGED SWAP: Ensure TTS is unloaded and Podcast is loaded
                # We do this here (inside the generator or before)
                # Since this runs in a thread, we should probably do the swap before yielding?
                # But generate_podcast_streaming does IO.
                # Let's do the swap in the main thread (here) before calling the generator thread
                pass 
                
            # Perform swap in main event loop before offloading
            if app.state.service:
                 app.state.service.unload()
            
            # Explicitly ensure podcast is loaded now
            service.ensure_loaded()

            def generator_wrapper():
                return service.generate_podcast_streaming(
                    num_speakers=num_speakers,
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    disable_voice_cloning=disable_cloning
                )

            queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def producer():
                try:
                    for item in generator_wrapper():
                        asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop) # Sentinel
                except Exception as e:
                    traceback.print_exc()
                    asyncio.run_coroutine_threadsafe(queue.put(e), loop)

            # Start producer thread
            threading.Thread(target=producer, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    await websocket.send_json({"type": "error", "message": str(item)})
                    break
                
                streaming_audio, complete_audio, log, vis = item
                
                # Send Log
                if log:
                    await websocket.send_json({"type": "log", "message": log})
                
                # Send Audio Chunk
                if streaming_audio:
                    sr, audio_np = streaming_audio
                    # Convert to PCM16 bytes
                    if audio_np.dtype == np.int16:
                        pcm = audio_np.tobytes()
                    else:
                        pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    await websocket.send_bytes(pcm)
                
                # Send Complete Audio status (not payload for now to keep it light)
                if complete_audio:
                     await websocket.send_json({"type": "complete"})
            
            await websocket.send_json({"type": "done"})

        except WebSocketDisconnect:
            service.stop_audio_generation()
        except Exception as e:
            traceback.print_exc()
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except: pass

@app.post("/tts_1_5b")
async def generate_tts_1_5b(
    text: str = Form(...),
    speaker: str = Form(...),
    cfg_scale: float = Form(1.3)
):
    try:
        # Ensure TTS 0.5B service is unloaded
        if hasattr(app.state, 'service') and app.state.service and hasattr(app.state.service, 'model') and app.state.service.model is not None:
            print("Unloading TTS 0.5B service...")
            # We don't have a direct unload method exposed on the service class shown in logs, 
            # but we can try to implement one or manually clean up if needed.
            # Looking at previous context, it seems PremiumStreamingService might not have an unload method in this version,
            # but the user objective said "unload() methods ... were confirmed to be functional".
            # Let's assume there is one or we invoke similar logic.
            if hasattr(app.state.service, 'unload'):
                app.state.service.unload()
            else:
                # Manual unload fallback
                del app.state.service.model
                del app.state.service.processor
                app.state.service.model = None
                app.state.service.processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
        podcast_service = app.state.podcast_service
        if not podcast_service:
             raise HTTPException(status_code=500, detail="Podcast service not initialized")
             
        podcast_service.ensure_loaded()
        
        sr, audio_np = podcast_service.generate_tts(text, speaker, cfg_scale)
        
        # Convert to WAV in memory
        import io
        import soundfile as sf
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sr, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
        
    except Exception as e:
        print(f"Error in TTS 1.5B: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone_voice")
async def clone_voice(
    file: UploadFile = File(...),
    language: str = Form("en"),
    gender: str = Form("man"),
    name: str = Form(...)
):
    try:
        # Validate inputs
        if not name or not name.strip():
             raise HTTPException(status_code=400, detail="Name is required")
             
        podcast_service = app.state.podcast_service
        if not podcast_service:
             raise HTTPException(status_code=500, detail="Podcast service not initialized")
             
        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-','_')).strip()
        safe_lang = language.lower().strip()
        safe_gender = gender.lower().strip()
        
        filename = f"{safe_lang}-{safe_name}-{safe_gender}.wav"
        save_path = os.path.join(podcast_service.voices_dir, filename)
        
        # Ensure dir exists (it should, but safety first)
        os.makedirs(podcast_service.voices_dir, exist_ok=True)
        
        # Save file
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Refresh voices
        podcast_service.refresh_voices()
        
        return {"status": "success", "message": f"Voice cloned successfully as {filename}", "voice_id": filename.replace('.wav', '')}
        
    except Exception as e:
        print(f"Error cloning voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cloned_voices")
async def get_cloned_voices():
    try:
        podcast_service = app.state.podcast_service
        if not podcast_service:
            raise HTTPException(status_code=500, detail="Podcast service not initialized")
        
        voices_dir = podcast_service.voices_dir
        if not os.path.exists(voices_dir):
            return {"voices": []}
        
        voices = []
        for filename in os.listdir(voices_dir):
            if filename.endswith('.wav'):
                parts = filename.replace('.wav', '').split('-')
                if len(parts) >= 3:
                    lang, name, gender = parts[0], '-'.join(parts[1:-1]), parts[-1]
                    voices.append({
                        "id": filename.replace('.wav', ''),
                        "name": name,
                        "language": lang,
                        "gender": gender,
                        "filename": filename
                    })
        
        return {"voices": voices}
        
    except Exception as e:
        print(f"Error getting cloned voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cloned_voices/{voice_id}")
async def delete_cloned_voice(voice_id: str):
    try:
        podcast_service = app.state.podcast_service
        if not podcast_service:
            raise HTTPException(status_code=500, detail="Podcast service not initialized")
        
        voices_dir = podcast_service.voices_dir
        filename = f"{voice_id}.wav"
        filepath = os.path.join(voices_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Voice not found")
        
        os.remove(filepath)
        
        # Refresh voices
        podcast_service.refresh_voices()
        
        return {"status": "success", "message": f"Voice {voice_id} deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting cloned voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/podcast/examples")
async def get_podcast_examples():
    if not app.state.podcast_service:
        return []
    return [{"index": i, "speakers": s[0], "preview": s[1][:50] + "..."} for i, s in enumerate(app.state.podcast_service.example_scripts)]

@app.get("/podcast/example/{index}")
async def get_podcast_example(index: int):
    if not app.state.podcast_service:
        raise HTTPException(status_code=404, detail="Service not initialized")
    
    scripts = app.state.podcast_service.example_scripts
    if 0 <= index < len(scripts):
        return {"speakers": scripts[index][0], "script": scripts[index][1]}
    else:
        raise HTTPException(status_code=404, detail="Example not found")

# ------------------------------------------------------------------------------
# OpenAI API Routes
# ------------------------------------------------------------------------------

@app.get("/v1/audio/voices")
async def list_openai_voices():
    """List available voices (OpenAI-compatible)"""
    service = app.state.openai_service
    service.refresh_voices()
    return VoicesResponse(voices=service.get_available_voices())

@app.get("/v1/audio/models")
async def list_openai_models():
    """List available TTS models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            }
        ]
    }

@app.post("/v1/audio/speech")
async def create_openai_speech(request: TTSRequest):
    """Generate speech from text (OpenAI-compatible)"""
    try:
        service = app.state.openai_service
        lock = app.state.openai_lock

        if lock.locked():
            raise HTTPException(status_code=429, detail="Service busy")

        async with lock:
            # Ensure loaded
            if service.model is None:
                service.load()

            # Validate
            if not request.input or not request.input.strip():
                raise HTTPException(status_code=400, detail="Input text is required")
            if len(request.input) > 4096:
                raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")
            supported_formats = ["wav", "pcm"]
            if request.response_format.lower() not in supported_formats:
                raise HTTPException(status_code=400, detail=f"Unsupported format. Supported: {supported_formats}")

            if request.stream:
                # Streaming response - always use PCM format for streaming
                def audio_generator():
                    for chunk in service.generate_speech_streaming(
                        text=request.input,
                        voice=request.voice,
                        cfg_scale=1.25
                    ):
                        yield chunk

                return StreamingResponse(
                    audio_generator(),
                    media_type="audio/pcm",
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.pcm"
                    }
                )
            else:
                # Non-streaming response
                audio = service.generate_speech(
                    text=request.input,
                    voice=request.voice,
                    cfg_scale=1.25
                )

                # Convert
                audio_bytes = convert_audio(audio, request.response_format)
                content_type = get_content_type(request.response_format)

                from fastapi import Response
                return Response(
                    content=audio_bytes,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
                    }
                )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/models/load")
async def load_openai_model():
    """Load the OpenAI TTS model"""
    try:
        service = app.state.openai_service
        lock = app.state.openai_lock

        if lock.locked():
            raise HTTPException(status_code=429, detail="Service busy - model loading in progress")

        async with lock:
            if service.model is None:
                print(f"[{get_timestamp()}] Loading OpenAI TTS model...")
                service.load()
                print(f"[{get_timestamp()}] OpenAI TTS model loaded successfully")
                return {"status": "success", "message": "Model loaded successfully"}
            else:
                return {"status": "already_loaded", "message": "Model is already loaded"}

    except Exception as e:
        print(f"[{get_timestamp()}] Error loading OpenAI TTS model: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
