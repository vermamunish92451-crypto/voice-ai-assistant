"""
Voice AI Assistant - Faster-Whisper (STT) + LLM + Kokoro-ONNX (TTS) - Python 3.14 compatible
Run: python app_py314.py
"""

import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf

# Auto-load .env if exists (so you don't need $env: every time)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
# Also try to load from .env file manually
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k,v = line.strip().split("=",1)
                if k and v and not os.getenv(k):
                    os.environ[k]=v.strip().strip('"').strip("'")

# ========== CONFIG ==========
WHISPER_MODEL = "base"
KOKORO_VOICE = "af_heart"  # af_heart, af_bella, am_michael
SAMPLE_RATE = 24000
LLM_MODE = "echo"

print("Loading models...")

# ========== 1. WHISPER (STT) ==========
from faster_whisper import WhisperModel
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print(f"✓ Whisper loaded: {WHISPER_MODEL}")

def transcribe(audio_path):
    if audio_path is None:
        return ""
    try:
        # language=None = auto-detect (don't use "auto")
        segments, info = whisper.transcribe(audio_path, language=None, beam_size=5, vad_filter=True)
        text = " ".join([s.text.strip() for s in segments]).strip()
        print(f"[STT] {text} | lang: {info.language}")
        return text
    except Exception as e:
        print(f"STT Error: {e}")
        return f"Error: {e}"

# ========== 2. LLM - SMART LOCAL FALLBACK (No API needed) ==========
def local_smart_reply(text):
    t = text.lower()
    if any(x in t for x in ["hello", "hi", "hey", "namaste", "hlo"]):
        return "Hello! I am your Voice AI Assistant built with Faster-Whisper and Kokoro. Kaise madad kar sakta hun?"
    if "your name" in t or "tumhara naam" in t:
        return "Mera naam Voice AI Assistant hai, mai Faster-Whisper se sunta hun aur Kokoro se bolta hun!"
    if "how are you" in t or "kaise ho" in t:
        return "Mai ekdum mast hun! Aap sunao, kya help chahiye?"
    if "what can you do" in t or "kya kar sakte" in t:
        return "Mai aapki awaz ko text me badal sakta hun, sawal ka jawab de sakta hun, aur apne jawab ko natural awaz me suna sakta hun!"
    if "bye" in t or "alvida" in t:
        return "Alvida! Phir milte hain, dhanyavaad!"
    if "thank" in t or "shukriya" in t:
        return "Aapka swagat hai! Aur kuch puchna hai?"
    # default - echo with AI touch
    return f"Aapne kaha: '{text}'. Ye bahut interesting hai! Is bare me aur batao, mai sun raha hun."

def get_llm_response(text, mode=LLM_MODE):
    if not text or text.strip() == "":
        return "Mujhe kuch sunai nahi diya, phir se bolo!"
    if mode == "echo":
        return local_smart_reply(text)
    elif mode == "openai":
        try:
            from openai import OpenAI
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key or api_key == "YOUR_KEY" or len(api_key) < 20:
                print("[LLM] No valid API key, using local smart reply")
                return local_smart_reply(text) + " (Tip: .env me GROQ_API_KEY dalo toh Groq ka real AI chalega)"
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            models_to_try = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "mixtral-8x7b-32768", "gemma2-9b-it"]
            last_err = None
            for m in models_to_try:
                try:
                    res = client.chat.completions.create(
                        model=m,
                        messages=[{"role":"system","content":"You are a helpful voice assistant, concise (2-3 sentences). Reply in same language as user."},{"role":"user","content":text}],
                        max_tokens=150
                    )
                    print(f"[LLM] Groq model used: {m}")
                    return res.choices[0].message.content
                except Exception as e2:
                    last_err = e2
                    if "404" in str(e2) or "model" in str(e2).lower():
                        print(f"[LLM] model {m} not found, trying next...")
                        continue
                    else:
                        raise
            print(f"[LLM] All Groq models failed: {last_err}, falling back to local")
            return local_smart_reply(text)
        except Exception as e:
            print(f"[LLM] Error: {e}, fallback to local")
            return local_smart_reply(text)
            return res.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}"
    return text

# ========== 3. KOKORO-ONNX (TTS) - Python 3.14 compatible ==========
try:
    from kokoro_onnx import Kokoro
    # Model files auto-download if not present
    # kokoro-onnx needs model files: kokoro-v1.0.onnx + voices.bin
    # We'll lazy download
    import urllib.request
    from pathlib import Path

    MODEL_PATH = Path("assets/kokoro-v1.0.onnx")
    VOICES_PATH = Path("assets/voices-v1.0.bin")
    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def ensure_model():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            print(f"Downloading Kokoro model... {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✓ Model downloaded")
        if not VOICES_PATH.exists():
            print(f"Downloading voices... {VOICES_URL}")
            urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)
            print("✓ Voices downloaded")

    ensure_model()
    kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    print(f"✓ Kokoro-ONNX loaded: voice={KOKORO_VOICE}")

    def tts_kokoro(text, voice=KOKORO_VOICE):
        if not text or not text.strip():
            return None
        try:
            # Split long text
            samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp.name, samples, sr)
            print(f"[TTS] {len(text)} chars -> {len(samples)/sr:.2f}s")
            return tmp.name
        except Exception as e:
            print(f"TTS Error: {e}")
            import traceback; traceback.print_exc()
            return None

except Exception as e:
    print(f"Kokoro-ONNX load failed: {e}")
    import traceback; traceback.print_exc()
    def tts_kokoro(text, voice=None):
        return None

# ========== 4. PIPELINE ==========
def voice_assistant(audio_path, llm_mode, voice_choice):
    if audio_path is None:
        return "", "Mic se bolo pehle!", None
    transcript = transcribe(audio_path)
    if not transcript:
        return "", "Kuch samajh nahi aaya", None
    reply = get_llm_response(transcript, mode=llm_mode)
    audio_out = tts_kokoro(reply, voice=voice_choice)
    return transcript, reply, audio_out

def tts_only(text, voice_choice):
    if not text.strip():
        return None
    return tts_kokoro(text, voice=voice_choice)

# ========== 5. UI ==========
with gr.Blocks(title="Voice AI Assistant") as app:
    gr.Markdown("# 🎙️ Voice AI Assistant (Python 3.14 - Kokoro-ONNX)")
    gr.Markdown("**Mic se bolo → Faster-Whisper → Smart AI → Kokoro bolega | No API needed!**")

    with gr.Tab("🎤 Voice Chat"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Bolo")
                llm_mode = gr.Radio(["echo", "openai"], value="echo", label="LLM Mode", info="echo=Local smart (no key needed) ✅ | openai=Groq AI (needs .env)")
                voice = gr.Dropdown(["af_heart","af_bella","af_sarah","am_michael","am_adam"], value="af_heart", label="Voice")
                btn = gr.Button("🚀 Send", variant="primary")
            with gr.Column():
                transcript = gr.Textbox(label="📝 Transcript", lines=3)
                reply = gr.Textbox(label="🤖 Reply", lines=4)
                audio_out = gr.Audio(label="🔊 Audio", autoplay=True)
        btn.click(voice_assistant, inputs=[audio_in, llm_mode, voice], outputs=[transcript, reply, audio_out])
        audio_in.stop_recording(voice_assistant, inputs=[audio_in, llm_mode, voice], outputs=[transcript, reply, audio_out])

    with gr.Tab("🔤 TTS Only"):
        tts_text = gr.Textbox(label="Text", value="Hello! I am your voice assistant built with Kokoro and Faster Whisper.", lines=3)
        tts_voice = gr.Dropdown(["af_heart","af_bella","am_michael"], value="af_heart", label="Voice")
        tts_btn = gr.Button("🔊 Generate")
        tts_audio = gr.Audio(label="Output", autoplay=True)
        tts_btn.click(tts_only, inputs=[tts_text, tts_voice], outputs=tts_audio)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
