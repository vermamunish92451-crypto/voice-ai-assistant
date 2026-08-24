# 🎙️ Voice AI Assistant — Faster-Whisper + Kokoro

> Mic se bolo → AI samjhe → Kokoro natural awaz me jawab de

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org)
[![Kokoro-82M](https://img.shields.io/badge/TTS-Kokoro--82M-green)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Faster-Whisper](https://img.shields.io/badge/STT-Faster--Whisper-orange)](https://github.com/SYSTRAN/faster-whisper)
[![Gradio](https://img.shields.io/badge/UI-Gradio-red)](https://gradio.app)

### ✨ Demo
<!-- Add your screen recording here: assets/demo.mp4 or GIF -->
`http://127.0.0.1:7860` — Voice Chat + TTS Only

### 🚀 Features
- 🎤 **STT:** Faster-Whisper `base` (4x faster than Whisper, VAD, auto language)
- 🗣️ **TTS:** Kokoro-82M via `kokoro-onnx` (Python 3.14 compatible, voices: `af_heart`, `am_michael`, `hf_alpha` Hindi)
- 🤖 **LLM:** Dual mode — `echo` (local smart fallback, no API needed) + `openai` (Groq: `llama-3.1/3.3`, `gpt-oss`, `mixtral`)
- 🔊 Gradio UI with mic, live transcript, reply & auto-play audio
- ⚡ Works offline (echo mode), 1-click start (`START.bat`)

### 🏗️ How it Works
```
Microphone → Faster-Whisper (audio → text) → LLM (text → reply) → Kokoro-ONNX (reply → audio) → Speaker
```

### 🛠️ Tech Stack
`faster-whisper` `kokoro-onnx` `onnxruntime` `gradio` `soundfile` `openai` (Groq)

### ⚡ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/voice-ai-assistant.git
cd voice-ai-assistant
pip install faster-whisper kokoro-onnx gradio soundfile onnxruntime openai python-dotenv
python app_py314.py
# open http://127.0.0.1:7860
```
Windows 1-click: double-click `START.bat`

**Groq (optional):** Create `.env` from `.env.example`:
```
GROQ_API_KEY=gsk_...
```
Get free key: https://console.groq.com/keys

### 🎛️ Config
Top of `app_py314.py`:
```python
WHISPER_MODEL = "base"  # small/medium = more accurate
KOKORO_VOICE = "af_heart"
```

### 📁 Structure
```
voice-ai-assistant/
├── app_py314.py     # Main app (Python 3.14)
├── app.py           # Original (Python 3.10-3.12, kokoro)
├── START.bat        # 1-click launch
├── .env.example     # API key template
└── assets/          # kokoro-v1.0.onnx + voices.bin (auto-download)
```

### 🙏 Credits
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Gradio](https://gradio.app)

Built with ❤️ for learning — Mini Project
