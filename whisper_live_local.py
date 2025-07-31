import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ssl
import asyncio
import threading
import queue
import time
import os
import re
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from faster_whisper import WhisperModel

# --- SSL fix for macOS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Settings ---
SAMPLE_RATE = 16000
BLOCK_SECONDS = 5
DEVICE_NAME = "BlackHole 2ch"
CONTEXT_SIZE = 3

# --- Load .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Queues ---
audio_queue = queue.Queue()
transcribe_queue = queue.Queue()
text_queue = asyncio.Queue()

# --- Context buffer ---
sentence_buffer = []

# --- Logger ---
def log(msg):
    print(f"[LOG {time.strftime('%H:%M:%S')}]: {msg}")

# --- Sentence splitter ---
def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+|\n+|— ')
    return sentence_endings.split(text.strip())

# --- Load faster-whisper model ---
log("🔁 Loading faster-whisper model...")
model = WhisperModel("small", compute_type="int8")
log("✅ faster-whisper model loaded.")

# --- Audio callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"⚠️ Audio status: {status}")
    audio_queue.put(indata.copy())

# --- OpenAI question checker ---
async def is_question_openai_async(text: str) -> bool:
    prompt = f"Decide if the following text is a question. Answer only 'yes' or 'no'.\n\nText: \"{text}\""
    try:
        log(f"→ GPT prompt: {text}")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You classify if the input text is a question."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0,
        )
        answer = response.choicess[0].message.content.strip().lower()
        log(f"← GPT response: {answer}")
        return answer.startswith("yes")
    except Exception as e:
        log(f"OpenAI API error: {e}")
        return False

# --- Text processing task ---
async def process_texts():
    while True:
        text = await text_queue.get()
        if text is None:
            break
        log(f"📄 Received transcription: {text}")
        sentences = split_into_sentences(text)
        log(f"✂️ Split into sentences: {sentences}")

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_buffer.append(sentence)
            if len(sentence_buffer) > CONTEXT_SIZE:
                sentence_buffer.pop(0)

            context_text = " ".join(sentence_buffer)
            log(f"🤔 Checking if question (context): {context_text}")
            is_question = await is_question_openai_async(context_text)

            if is_question:
                print(f"\n❓ Question detected: {sentence}\n")

# --- Transcription worker ---
def transcribe_worker():
    while True:
        audio_chunk = transcribe_queue.get()
        if audio_chunk is None:
            break
        log("🔊 Transcribing audio chunk...")
        segments, _ = model.transcribe(audio_chunk, language="uk")
        text = " ".join([seg.text for seg in segments]).strip()
        if text:
            log(f"📝 Transcription result: {text}")
            asyncio.run_coroutine_threadsafe(text_queue.put(text), async_loop)

# --- Async loop runner ---
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async_loop = asyncio.new_event_loop()
threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True).start()
asyncio.run_coroutine_threadsafe(process_texts(), async_loop)

# --- Find device index ---
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if DEVICE_NAME in dev['name']:
        device_index = idx
        break
if device_index is None:
    raise RuntimeError(f"Device '{DEVICE_NAME}' not found")
log(f"🎧 Using device: {DEVICE_NAME} (index {device_index})")

# --- Start transcription thread ---
threading.Thread(target=transcribe_worker, daemon=True).start()

# --- Main recording loop ---
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    device=device_index
):
    buffer = np.empty((0, 1), dtype='float32')
    last_time = time.time()
    log("✅ Listening... Press Ctrl+C to stop.")

    try:
        while True:
            data = audio_queue.get()
            buffer = np.append(buffer, data, axis=0)
            if time.time() - last_time >= BLOCK_SECONDS:
                last_time = time.time()
                if len(buffer) == 0:
                    continue
                audio_chunk = buffer.flatten()
                buffer = np.empty((0, 1), dtype='float32')
                transcribe_queue.put(audio_chunk)
    except KeyboardInterrupt:
        log("⏹️ Stopping...")
        transcribe_queue.put(None)
        asyncio.run_coroutine_threadsafe(text_queue.put(None), async_loop)
        async_loop.call_soon_threadsafe(async_loop.stop)
