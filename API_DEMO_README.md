# VibeVoice API Demo

A simple HTML/JavaScript demo for testing the VibeVoice OpenAI-compatible TTS API.

## Features

- 🎤 **Voice Selection**: Browse and select from available voices
- 📝 **Text Input**: Enter text to convert to speech
- 🎵 **Audio Generation**: Generate speech using the API
- 📥 **Download**: Download generated audio files
- 🌊 **Streaming Support**: Option to enable streaming responses

## API Endpoints Tested

- `GET /v1/audio/voices` - List available voices
- `GET /v1/audio/models` - List available models
- `POST /v1/audio/speech` - Generate speech from text

## Setup

1. **Start the VibeVoice server:**
   ```bash
   python -m uvicorn vibevoice_premium.app:app --host 0.0.0.0 --port 7860
   ```

2. **Open the API demo:**
   - Visit: `http://localhost:7860/api-demo`
   - This serves the demo HTML with proper CORS headers

## Alternative: Open HTML directly

If you prefer to open the HTML file directly:

1. Start a simple HTTP server in the project directory:
   ```bash
   python -m http.server 8000
   ```

2. Visit: `http://localhost:8000/api_demo.html`

**Note:** Opening `api_demo.html` directly in the browser (file:// protocol) won't work due to CORS restrictions. You must serve it via HTTP.

## API Compatibility

This demo uses the OpenAI-compatible TTS API with the following features:

- **Voices**: alloy, echo, fable, onyx, nova, shimmer (mapped to VibeVoice voices)
- **Models**: tts-1, tts-1-hd
- **Formats**: WAV, PCM
- **Streaming**: Supported for real-time audio generation

## Usage

1. Select a voice from the grid or dropdown
2. Enter text to synthesize
3. Choose model and audio format
4. Enable streaming if desired
5. Click "Generate Speech"
6. Play or download the generated audio

## Python Test Script

Also included: `test_vibevoice_api.py` - A Python script for testing the API programmatically.

```bash
python test_vibevoice_api.py
```