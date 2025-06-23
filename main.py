import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import sounddevice as sd
import numpy as np
import whisper
import time
import queue

# далі ваш код...


# Налаштування
SAMPLE_RATE = 16000
BLOCK_SECONDS = 5  # кожні 5 сек обробка

# Черга для аудіоданих
audio_queue = queue.Queue()

# Завантаження Whisper-моделі
print("🔁 Завантаження моделі Whisper...")
model = whisper.load_model("small")
print("✅ Модель завантажено. Слухаємо...")


# Функція зворотного виклику
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(indata.copy())


# Пошук індексу пристрою "BlackHole 2ch"
device_name = "BlackHole 2ch"
device_index = None
for idx, dev in enumerate(sd.query_devices()):
    if device_name in dev['name']:
        device_index = idx
        break

if device_index is None:
    raise RuntimeError("Не знайдено пристрій BlackHole 2ch")

# Аудіо потік
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

            # обробка кожні N секунд
            if time.time() - last_time > BLOCK_SECONDS:
                last_time = time.time()
                if len(buffer) == 0:
                    continue

                # перетворення формату
                audio = buffer.flatten()
                buffer = np.empty((0, 1), dtype='float32')

                # збереження аудіо тимчасово в .wav
                print("🔊 Розпізнавання мовлення...")
                result = model.transcribe(audio, fp16=False, language='uk')  # або 'uk'
                print(f"[🗣️]: {result['text'].strip()}\n")

        except KeyboardInterrupt:
            print("\n⏹️ Завершення...")
            break
