import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import sounddevice as sd
import numpy as np
import whisper
import time
import queue

# Settings
SAMPLE_RATE = 16000
BLOCK_SECONDS = 5  # process every 5 seconds

# Queue for audio data
audio_queue = queue.Queue()

# Loading the Whisper model
print("🔁 Loading Whisper model...")
model = whisper.load_model("small")
print("✅ Model loaded. Listening...")


# Callback function
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(indata.copy())


# Find device index for "BlackHole 2ch"
device_name = "BlackHole 2ch"
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if device_name in dev['name']:
        device_index = idx
        break

if device_index is None:
    raise RuntimeError("BlackHole 2ch device not found")

# Audio stream
with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=0,
        device=device_index
):
    buffer = np.empty((0, 1), dtype='float32')
    last_time = time.time()

    while True:
        try:
            data = audio_queue.get()
            buffer = np.append(buffer, data, axis=0)

            # processing every N seconds
            if time.time() - last_time > BLOCK_SECONDS:
                last_time = time.time()
                if len(buffer) == 0:
                    continue

                # format conversion
                audio = buffer.flatten()
                buffer = np.empty((0, 1), dtype='float32')

                # temporarily save audio to .wav
                print("🔊 Speech recognition...")
                result = model.transcribe(audio, fp16=False, language='uk')  # or 'uk'
                print(f"[🗣️]: {result['text'].strip()}\n")

        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
            break
