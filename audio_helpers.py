import logging
import select
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from config import (
    KOKORO_MODEL_FILE,
    KOKORO_VOICES_FILE,
    RECORD_SAMPLERATE,
    RECORD_SECONDS,
    TTS_SPEED,
    TTS_VOICE,
    WHISPER_MODEL,
)

log = logging.getLogger(__name__)


def load_tts():
    from kokoro_onnx import Kokoro

    for fpath in (KOKORO_MODEL_FILE, KOKORO_VOICES_FILE):
        if not fpath.exists():
            log.error(
                f"'{fpath.name}' not found. Download it from the kokoro-onnx GitHub releases."
            )
            raise FileNotFoundError(fpath)

    log.info("Loading Kokoro TTS...")
    tts = Kokoro(str(KOKORO_MODEL_FILE), str(KOKORO_VOICES_FILE))
    log.info("Kokoro TTS ready.")
    return tts


def speak(tts, text: str) -> None:
    log.info(f"Speaking: {text[:80]}...")
    try:
        samples, sample_rate = tts.create(
            text,
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            lang="en-us",
        )
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        log.error(f"TTS error: {e}")


def load_whisper():
    try:
        import whisper
    except ImportError:
        log.error("Whisper not installed. Run: pip install openai-whisper")
        raise

    log.info(f"Loading Whisper '{WHISPER_MODEL}'...")
    model = whisper.load_model(WHISPER_MODEL)
    log.info("Whisper STT ready.")
    return model


def record_audio(
    seconds: int = RECORD_SECONDS,
    samplerate: int = RECORD_SAMPLERATE,
) -> Optional[np.ndarray]:
    """Record from the microphone until Enter is pressed or timeout expires."""
    import sys

    print(f"\n  Recording up to {seconds}s... (press Enter to stop early)")
    frames = []
    stop_event = threading.Event()

    def _record():
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
            while not stop_event.is_set():
                data, _ = stream.read(1024)
                frames.append(data.copy())

    t = threading.Thread(target=_record, daemon=True)
    t.start()

    try:
        ready, _, _ = select.select([sys.stdin], [], [], seconds)
        if ready:
            sys.stdin.readline()
    except Exception:
        time.sleep(seconds)

    stop_event.set()
    t.join(timeout=2)

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    print(f"  Recorded {len(audio) / samplerate:.1f}s")
    return audio


def transcribe(whisper_model, audio: np.ndarray) -> Optional[str]:
    log.info("Transcribing with Whisper...")
    try:
        result = whisper_model.transcribe(audio, fp16=False, language="en")
        text = result["text"].strip()
        log.info(f"Transcript: {text}")
        return text
    except Exception as e:
        log.error(f"Whisper error: {e}")
        return None
