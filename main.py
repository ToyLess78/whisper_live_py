import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ssl
import asyncio
import threading
import numpy as np
import sounddevice as sd
import collections
import wave
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
from faster_whisper import WhisperModel
from pynput import keyboard as pynput_keyboard

# --- SSL fix for macOS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Settings ---
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SECONDS = 10
DEVICE_NAME = "BlackHole 2ch"
BUFFER_SIZE = SAMPLE_RATE * DURATION_SECONDS

# --- Load .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Logger ---
def log(msg):
    print(f"[LOG {time.strftime('%H:%M:%S')}]: {msg}")

# --- Buffer for last N seconds of audio ---
audio_buffer = collections.deque(maxlen=BUFFER_SIZE)

# --- Audio callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"⚠️ Audio status: {status}")
    audio_buffer.extend(indata[:, 0])

# --- Save buffer to WAV ---
def save_buffer_to_wav(filename="recorded_chunk.wav"):
    audio_np = np.array(audio_buffer, dtype=np.float32)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    log(f"💾 Saved last {DURATION_SECONDS}s to {filename}")
    return filename

# --- Transcribe and ask GPT ---
async def handle_question_from_audio():
    filename = save_buffer_to_wav()
    segments, _ = model.transcribe(filename, language="uk")
    text = " ".join([seg.text for seg in segments]).strip()
    log(f"📝 Transcribed: {text}")
    if not text:
        print("⚠️ No text recognized.")
        return

    prompt = f"Please answer this interview-related question as if you are a developer: \"{text}\""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful developer being interviewed."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        print("\n🤖 GPT Answer:\n", answer, "\n")
    except Exception as e:
        log(f"❌ OpenAI error: {e}")

# --- Load model ---
log("🔁 Loading faster-whisper model...")
model = WhisperModel("tiny", compute_type="int8")
log("✅ faster-whisper model loaded.")

# --- Find device index ---
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if DEVICE_NAME in dev['name']:
        device_index = idx
        break
if device_index is None:
    raise RuntimeError(f"Device '{DEVICE_NAME}' not found")
log(f"🎧 Using device: {DEVICE_NAME} (index {device_index})")

# --- Keyboard listener ---
def on_press(key):
    try:
        if key.char == 's':
            log("🎯 's' pressed: analyzing last 10s...")
            asyncio.run(handle_question_from_audio())
    except AttributeError:
        pass  # special keys

listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()

# --- Start recording ---
log("✅ Listening... Press 's' to send last 10s to GPT or Ctrl+C to stop.")
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        device=device_index
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    log("⏹️ Stopping...")
