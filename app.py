"""
Voice AI Assistant - Faster-Whisper (STT) + LLM + Kokoro (TTS)
Run: python app.py
- Mic se bolo -> Whisper transcribe -> LLM jawab -> Kokoro bolega
"""
import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf

# ========== CONFIG ==========
WHISPER_MODEL = "base"  # base / small / medium | base = fast, small = balanced
KOKORO_VOICE = "af_heart"  # af_heart, af_bella, am_michael, hf_alpha (hindi)
KOKORO_LANG = "a"  # a=American English, h=Hindi, b=British
LLM_MODE = "echo"  # echo | openai | ollama
SAMPLE_RATE = 24000

print("Loading models... (first time 1-2GB download hoga)")

# ========== 1. WHISPER (STT) ==========
from faster_whisper import WhisperModel
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print(f"✓ Whisper loaded: {WHISPER_MODEL}")

def transcribe(audio_path):
    if audio_path is None:
        return ""
    try:
        segments, info = whisper.transcribe(audio_path, language="auto", beam_size=5, vad_filter=True)
        text = " ".join([s.text.strip() for s in segments]).strip()
        print(f"[STT] {text} | lang: {info.language} ({info.language_probability:.2f})")
        return text
    except Exception as e:
        print(f"STT Error: {e}")
        return f"Error: {e}"

# ========== 2. LLM (Response) ==========
# Option A: Echo (no API needed) - default
# Option B: OpenAI / Groq - uncomment and set API key
# Option C: Ollama local - pip install ollama

def get_llm_response(text, mode=LLM_MODE):
    if not text or text.strip() == "":
        return "Mujhe kuch sunai nahi diya, phir se bolo please!"

    if mode == "echo":
        # Simple demo without LLM API
        return f"Aapne kaha: '{text}'. Mai aapka Voice Assistant hu! Aap kaise ho?"

    elif mode == "openai":
        try:
            from openai import OpenAI
            # Groq free API example - change if using OpenAI
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
            if api_key == "YOUR_API_KEY_HERE":
                return "API key set nahi hai. .env me GROQ_API_KEY dalo."
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Reply in same language user speaks, concise (2-3 sentences)."},
                    {"role": "user", "content": text}
                ],
                max_tokens=150
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}. Echo mode use kar raha hu: Aapne kaha '{text}'"

    elif mode == "ollama":
        try:
            import ollama
            res = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": text}])
            return res['message']['content']
        except Exception as e:
            return f"Ollama Error: {e}"

    return text

# ========== 3. KOKORO (TTS) ==========
try:
    from kokoro import KPipeline
    # espeak-ng required for some setups - if not installed, fallback still works for English
    kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
    print(f"✓ Kokoro loaded: lang={KOKORO_LANG}, voice={KOKORO_VOICE}")

    def tts_kokoro(text):
        if not text or not text.strip():
            return None
        try:
            # Kokoro has limit ~500 chars per chunk, auto chunks
            # generator yields (grapheme, phoneme, audio)
            audios = []
            generator = kokoro_pipeline(text, voice=KOKORO_VOICE)
            for i, (gs, ps, audio) in enumerate(generator):
                print(f"[TTS] chunk {i}: {gs[:50]}... -> {len(audio)/24000:.2f}s")
                audios.append(audio)
            if not audios:
                return None
            full_audio = np.concatenate(audios)
            # Save to temp wav for Gradio
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp.name, full_audio, SAMPLE_RATE)
            return tmp.name
        except Exception as e:
            print(f"TTS Error: {e}")
            import traceback; traceback.print_exc()
            return None

except Exception as e:
    print(f"Kokoro load failed: {e} - TTS disabled")
    def tts_kokoro(text):
        return None

# ========== 4. MAIN PIPELINE ==========
def voice_assistant(audio_path, llm_mode, voice_choice):
    global KOKORO_VOICE, kokoro_pipeline
    if voice_choice != KOKORO_VOICE:
        KOKORO_VOICE = voice_choice
        print(f"Voice changed to {KOKORO_VOICE}")

    if audio_path is None:
        return "", "Mic se kuch bolo pehle!", None

    # 1. STT
    transcript = transcribe(audio_path)
    if not transcript:
        return "", "Kuch samajh nahi aaya, tez ya saaf bolo", None

    # 2. LLM
    reply = get_llm_response(transcript, mode=llm_mode)

    # 3. TTS
    audio_out = tts_kokoro(reply)

    return transcript, reply, audio_out

def text_to_speech_only(text, voice_choice):
    if not text.strip():
        return None
    global KOKORO_VOICE
    KOKORO_VOICE = voice_choice
    return tts_kokoro(text)

# ========== 5. GRADIO UI ==========
with gr.Blocks(title="Voice AI Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎙️ Voice AI Assistant
    **Faster-Whisper (STT) + LLM + Kokoro (TTS) | Mic se bolo → AI jawab dega → Kokoro bolega**
    """)

    with gr.Tab("🎤 Voice Chat"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Yahan bolo (Mic press karo)")
                llm_mode = gr.Radio(["echo", "openai", "ollama"], value="echo", label="LLM Mode", info="echo=free demo, openai=Groq/OpenAI API, ollama=local")
                voice = gr.Dropdown(["af_heart", "af_bella", "af_sarah", "am_michael", "am_adam", "hf_alpha", "hf_beta"], value="af_heart", label="Kokoro Voice")
                btn = gr.Button("🚀 Send", variant="primary")
                gr.Markdown("**Tip:** `openai` ke liye `GROQ_API_KEY` env variable set karo. `ollama` ke liye `ollama run llama3.2:1b` chalao")
            with gr.Column():
                transcript = gr.Textbox(label="📝 Aapne kaha (Transcript)", lines=3)
                reply = gr.Textbox(label="🤖 AI ka Jawab", lines=4)
                audio_out = gr.Audio(label="🔊 AI ki Awaz (Kokoro)", autoplay=True)

        btn.click(voice_assistant, inputs=[audio_in, llm_mode, voice], outputs=[transcript, reply, audio_out])
        audio_in.stop_recording(voice_assistant, inputs=[audio_in, llm_mode, voice], outputs=[transcript, reply, audio_out])

    with gr.Tab("🔤 Text to Speech Only"):
        gr.Markdown("Sirf text se audio banao - Whisper ki zarurat nahi")
        with gr.Row():
            tts_text = gr.Textbox(label="Text likho", value="Hello! I am your voice assistant built with Kokoro and Faster Whisper.", lines=3)
            tts_voice = gr.Dropdown(["af_heart", "af_bella", "am_michael", "hf_alpha"], value="af_heart", label="Voice")
        tts_btn = gr.Button("🔊 Generate Audio")
        tts_audio = gr.Audio(label="Output Audio", autoplay=True)
        tts_btn.click(text_to_speech_only, inputs=[tts_text, tts_voice], outputs=tts_audio)

    gr.Markdown("""
    ### Setup Tips
    - **Windows:** `espeak-ng` install karo agar Kokoro error de: https://github.com/espeak-ng/espeak-ng/releases -> .msi install
    - **Models:** Pehli baar 1-2GB download hoga, agla run fast hoga
    - **Faster:** Whisper `base` fast hai, `small` accurate hai - `WHISPER_MODEL` change karo
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
