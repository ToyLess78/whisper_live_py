import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ssl
import asyncio
import threading
import numpy as np
import sounddevice as sd
import collections
import io
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
DURATION_10S = 10
DURATION_5S = 5
DEVICE_NAME = "BlackHole 2ch"
# Buffer for maximum duration (10 seconds)
BUFFER_SIZE = SAMPLE_RATE * DURATION_10S

# --- Load .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Logger ---
def log(msg):
    print(f"[LOG {time.strftime('%H:%M:%S')}]: {msg}")

# --- Buffer for last N seconds of audio ---
audio_buffer = collections.deque(maxlen=BUFFER_SIZE)

# --- Processing state ---
is_processing = False
processing_lock = threading.Lock()

# --- Audio callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"⚠️ Audio status: {status}")
    audio_buffer.extend(indata[:, 0])

# --- Convert buffer to numpy array (in memory) ---
def get_audio_data(duration_seconds=10):
    """Convert buffer to numpy array without saving to file"""
    # Calculate how many samples we need for the specified duration
    samples_needed = SAMPLE_RATE * duration_seconds

    # Get the required number of samples from the end of the buffer
    if len(audio_buffer) >= samples_needed:
        audio_data = list(audio_buffer)[-samples_needed:]
    else:
        # If buffer doesn't have enough data, use all available
        audio_data = list(audio_buffer)

    audio_np = np.array(audio_data, dtype=np.float32)
    return audio_np

# --- Transcribe and ask GPT (optimized) ---
async def handle_question_from_audio(duration_seconds=10):
    global is_processing

    with processing_lock:
        if is_processing:
            log("⚠️ Already processing, ignoring request")
            return
        is_processing = True

    try:
        log(f"🔄 Processing last {duration_seconds}s of audio...")

        # Get audio data directly from buffer (no file I/O)
        audio_data = get_audio_data(duration_seconds)

        if len(audio_data) == 0:
            log("⚠️ No audio data in buffer")
            return

        # Create tasks for parallel execution
        transcription_task = asyncio.create_task(transcribe_audio(audio_data))

        # Wait for transcription
        text = await transcription_task

        if not text or len(text.strip()) < 3:
            log("⚠️ No meaningful text recognized")
            return

        log(f"📝 Transcribed ({duration_seconds}s): {text}")

        # Get GPT response
        await get_gpt_response(text)

    except Exception as e:
        log(f"❌ Processing error: {e}")
    finally:
        with processing_lock:
            is_processing = False

async def transcribe_audio(audio_data):
    """Transcribe audio data directly without file I/O"""
    def _transcribe():
        # Use WhisperModel with audio data directly
        segments, _ = model.transcribe(audio_data, language="uk")
        return " ".join([seg.text for seg in segments]).strip()

    # Run transcription in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe)

def format_text_with_wrap(text, width=60):
    """Format text with word wrapping at specified width"""
    import textwrap

    # Split text into paragraphs (preserve existing line breaks)
    paragraphs = text.split('\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if paragraph.strip():  # Skip empty lines
            # Wrap each paragraph
            wrapped = textwrap.fill(
                paragraph.strip(),
                width=width,
                break_long_words=False,
                break_on_hyphens=False
            )
            formatted_paragraphs.append(wrapped)
        else:
            formatted_paragraphs.append('')  # Preserve empty lines

    return '\n'.join(formatted_paragraphs)

async def get_gpt_response(text):
    """Get GPT response with optimized settings"""
    prompt = f"Please answer this interview-related question as if you are a developer: \"{text}\""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Faster and cheaper model
            messages=[
                {"role": "system", "content": "You are a helpful developer being interviewed. Be concise but informative."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,  # Reduced for faster response
            stream=False
        )
        answer = response.choices[0].message.content.strip()

        # Format the answer with word wrapping
        formatted_answer = format_text_with_wrap(answer, width=60)

        print(f"\n🤖 GPT Answer:")
        print(formatted_answer)
        print()  # Empty line after answer

    except Exception as e:
        log(f"❌ OpenAI error: {e}")

# --- Load model with optimizations ---
log("🔁 Loading faster-whisper model...")
# Use base model for better accuracy/speed balance, or tiny for maximum speed
model = WhisperModel(
    "base",  # Changed from "tiny" for better accuracy
    device="cpu",  # Explicit CPU usage
    compute_type="int8",
    num_workers=2  # Parallel processing
)
log("✅ faster-whisper model loaded.")

# --- Voice Activity Detection (optional optimization) ---
def has_speech(audio_data, threshold=0.01, min_duration=1.0):
    """Simple VAD to skip processing silence"""
    if len(audio_data) < SAMPLE_RATE * min_duration:
        return False

    # Calculate RMS energy
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms > threshold

# --- Find device index ---
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if DEVICE_NAME in dev['name']:
        device_index = idx
        break
if device_index is None:
    raise RuntimeError(f"Device '{DEVICE_NAME}' not found")
log(f"🎧 Using device: {DEVICE_NAME} (index {device_index})")

# --- Keyboard listener with debouncing ---
last_press_time = 0
DEBOUNCE_TIME = 1.0  # seconds

def on_press(key):
    global last_press_time
    try:
        current_time = time.time()
        duration = None

        # Check which key was pressed
        if key.char == 's':
            duration = DURATION_10S
            log("🎯 's' pressed: analyzing last 10s...")
        elif key.char == 'd':
            duration = DURATION_5S
            log("🎯 'd' pressed: analyzing last 5s...")

        if duration is not None:
            if current_time - last_press_time < DEBOUNCE_TIME:
                log("⚠️ Key press too fast, ignoring")
                return

            last_press_time = current_time

            # Run in background thread to avoid blocking
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(handle_question_from_audio(duration))
                loop.close()

            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()

    except AttributeError:
        pass  # special keys

listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()

# --- Preload model (warm-up) ---
log("🔥 Warming up model...")
dummy_audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.001
try:
    model.transcribe(dummy_audio)
    log("✅ Model warmed up")
except:
    log("⚠️ Model warm-up failed, continuing...")

# --- Start recording ---
log("✅ Listening... Press 's' for last 10s or 'd' for last 5s, Ctrl+C to stop.")
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        device=device_index,
        blocksize=1024  # Smaller blocks for lower latency
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    log("⏹️ Stopping...")
    listener.stop()
