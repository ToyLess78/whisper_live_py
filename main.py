import ssl
import asyncio
import threading
import queue
import time
import os
import re
import numpy as np
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- SSL fix для macOS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Налаштування ---
SAMPLE_RATE = 16000
BLOCK_SECONDS = 5
DEVICE_NAME = "BlackHole 2ch"

# --- Завантаження .env змінних ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Черги ---
audio_queue = queue.Queue()
transcribe_queue = queue.Queue()
text_queue = asyncio.Queue()

# --- Проста розбивка на речення ---
def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    return sentence_endings.split(text.strip())

# --- Логи з часом ---
def log(msg):
    print(f"[LOG {time.strftime('%H:%M:%S')}]: {msg}")

# --- Завантаження моделі Whisper ---
log("🔁 Loading Whisper model...")
model = whisper.load_model("small")  # або 'tiny' для швидкості
log("✅ Whisper model loaded.")

# --- Callback для аудіо потоку ---
def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"⚠️ Audio status: {status}")
    audio_queue.put(indata.copy())

# --- Класифікація чи є речення питанням ---
async def is_question_openai_async(text: str) -> bool:
    prompt = f"Decide if the following text is a question. Answer only 'yes' or 'no'.\n\nText: \"{text}\""
    try:
        log(f"→ GPT prompt: {text}")
        response = await client.chat.completions.create(
            model="gpt-4o",  # або "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You classify if the input text is a question."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        log(f"← GPT response: {answer}")
        return answer.startswith("yes")
    except Exception as e:
        log(f"OpenAI API error: {e}")
        return False

# --- Обробка текстів асинхронно ---
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
            log(f"🤔 Checking if question: {sentence}")
            is_question = await is_question_openai_async(sentence)
            if is_question:
                print(f"\n❓ Question detected: {sentence}\n")

# --- Робітник для Whisper ---
def transcribe_worker():
    while True:
        audio_chunk = transcribe_queue.get()
        if audio_chunk is None:
            break
        log("🔊 Transcribing audio chunk...")
        result = model.transcribe(audio_chunk, fp16=False, language="uk")
        text = result["text"].strip()
        if text:
            log(f"📝 Transcription result: {text}")
            asyncio.run_coroutine_threadsafe(text_queue.put(text), async_loop)

# --- Асинхронний цикл у фоновому потоці ---
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async_loop = asyncio.new_event_loop()
threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True).start()
asyncio.run_coroutine_threadsafe(process_texts(), async_loop)

# --- Знаходження аудіо-пристрою ---
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if DEVICE_NAME in dev['name']:
        device_index = idx
        break
if device_index is None:
    raise RuntimeError(f"Device '{DEVICE_NAME}' not found")
log(f"🎧 Using device: {DEVICE_NAME} (index {device_index})")

# --- Запуск потоку транскрипції ---
threading.Thread(target=transcribe_worker, daemon=True).start()

# --- Основний цикл прослуховування ---
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
