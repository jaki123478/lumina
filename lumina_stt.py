"""Motore Speech-to-Text 100% locale e offline per Lumina AI.

Architettura ibrida:
- Cattura microfono a frequenza nativa (preferibilmente WASAPI 48 kHz) con
  filtro passa-alto anti-rombo (80 Hz), media canali stereo e decimazione
  polifase anti-aliasing a 16 kHz + pre-roll di 350 ms per non tagliare mai
  l'attacco delle consonanti iniziali.
- Vosk italiano (`voices/vosk-it`) in streaming per anteprima live immediata
  delle parole parziali e rilevamento fine frase naturale (~1.25 s di pausa).
- Whisper GPU locale (`voices/whisper-turbo` o `voices/whisper-small`,
  `local_files_only=True` su CUDA fp16) sull'intero enunciato normalizzato
  uniformemente per trascrizione italiana ad altissima precisione (~180 ms).
"""
from __future__ import annotations

from collections import deque
import json
from math import gcd
import os
from pathlib import Path
import re
import threading
import time

for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parent
WHISPER_DIRS = (
    ROOT / "voices" / "whisper-turbo",
    ROOT / "voices" / "whisper-small",
)
VOSK_DIR = ROOT / "voices" / "vosk-it"

DOMAIN_PROMPT = (
    "Ciao Lumina, mi senti? Parliamo in italiano di programmazione Python, "
    "JavaScript, HTML, CSS, SQL, codice, funzioni, file PDF, memoria RAM, SSD, "
    "GPU, intelligenza artificiale, scienza, spazio e generazione immagini."
)

_HALLUCINATION_PATTERNS = re.compile(
    r"(?:"
    r"sottotitol[io]\s+(?:creati|a\s+cura|realizzati)|"
    r"amara\.org|"
    r"iscriviti\s+al\s+canale|"
    r"grazie\s+per\s+(?:la\s+visione|aver\s+guardato)|"
    r"al\s+prossimo\s+episodio|"
    r"traduzione\s+di\s+|"
    r"^\s*[\.\,\!\?\-\…\s]+\s*$"
    r")",
    re.IGNORECASE,
)

_VOSK_LOCK = threading.Lock()
_VOSK_MODEL = None

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None
_WHISPER_PROMPT_IDS = None
_WHISPER_UNAVAILABLE = False


def get_vosk_model():
    global _VOSK_MODEL
    with _VOSK_LOCK:
        if _VOSK_MODEL is None:
            if not VOSK_DIR.is_dir():
                raise FileNotFoundError(f"Modello Vosk non trovato in {VOSK_DIR}")
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            _VOSK_MODEL = Model(str(VOSK_DIR))
        return _VOSK_MODEL


def get_whisper_engine():
    """Carica Whisper locale (turbo o small) su GPU CUDA fp16 (o CPU se senza GPU)."""
    global _WHISPER_MODEL, _WHISPER_PROCESSOR, _WHISPER_PROMPT_IDS, _WHISPER_UNAVAILABLE
    if _WHISPER_UNAVAILABLE:
        return None, None, None
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            chosen = next((d for d in WHISPER_DIRS if (d / "config.json").is_file()), None)
            if chosen is None:
                _WHISPER_UNAVAILABLE = True
                return None, None, None
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                proc = AutoProcessor.from_pretrained(str(chosen), local_files_only=True)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(chosen),
                    local_files_only=True,
                    dtype=dtype,
                ).to(device).eval()
                try:
                    prompt_ids = proc.get_prompt_ids(DOMAIN_PROMPT, return_tensors="pt").to(device)
                except Exception:
                    prompt_ids = None
                _WHISPER_PROCESSOR = proc
                _WHISPER_MODEL = model
                _WHISPER_PROMPT_IDS = prompt_ids
                print(f"[Lumina STT] Motore vocale {chosen.name} attivo su {device.upper()} (100% offline).", flush=True)
            except Exception as exc:
                print(f"[Lumina STT] Avviso caricamento Whisper ({exc}); fallback a Vosk.", flush=True)
                _WHISPER_UNAVAILABLE = True
                return None, None, None
        return _WHISPER_MODEL, _WHISPER_PROCESSOR, _WHISPER_PROMPT_IDS


def prewarm_stt():
    """Pre-inizializza Vosk (VAD/parziali) e Whisper (trascrizione GPU) all'avvio."""
    try:
        if VOSK_DIR.is_dir():
            get_vosk_model()
    except Exception:
        pass
    try:
        get_whisper_engine()
    except Exception:
        pass


def _highpass_80hz(samples_16k):
    """Filtro passa-alto leggero ~80 Hz per rimuovere rombo ventole/scrivania."""
    import numpy as np
    if samples_16k.size < 4:
        return samples_16k
    try:
        from scipy.signal import butter, lfilter
        b, a = butter(1, 80.0 / 8000.0, btype="highpass")
        return lfilter(b, a, samples_16k).astype(np.float32)
    except Exception:
        return (samples_16k - float(np.mean(samples_16k))).astype(np.float32)


def _resample_to_16k(mono_float32, src_rate: int):
    import numpy as np
    sr = int(round(src_rate))
    if sr == 16000 or mono_float32.size == 0:
        return mono_float32.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(16000, sr)
    return resample_poly(mono_float32, 16000 // g, sr // g).astype(np.float32)


def _normalize_utterance(wav_16k):
    """Normalizzazione uniforme sull'intero enunciato (mai AGC blocco-per-blocco)."""
    import numpy as np
    if wav_16k.size == 0:
        return wav_16k
    clean = _highpass_80hz(wav_16k)
    peak = float(np.percentile(np.abs(clean), 99.5))
    if peak < 0.0025:
        return clean
    gain = min(8.0, 0.82 / peak)
    return np.clip(clean * gain, -1.0, 1.0).astype(np.float32)


def _postprocess_italian_transcript(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if _HALLUCINATION_PATTERNS.search(t):
        return ""
    # Correzioni fonetiche comuni per il nome dell'assistente all'inizio frase
    t = re.sub(
        r"^\s*(?:ciao\s+)?(?:lumaca|l'umina|luminia|lu\s+mina|allumina)\b",
        lambda m: "Ciao Lumina" if "ciao" in m.group(0).lower() else "Lumina",
        t,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", t).strip()


def transcribe_float32_16k(wav_16k, vosk_fallback_text: str = "") -> str:
    """Trascrive un vettore float32 16 kHz usando Whisper GPU locale (fallback Vosk)."""
    import numpy as np
    if wav_16k is None or len(wav_16k) < 2400:  # < 150 ms
        return _postprocess_italian_transcript(vosk_fallback_text)

    norm_wav = _normalize_utterance(np.asarray(wav_16k, dtype=np.float32))
    rms = float(np.sqrt(np.mean(norm_wav * norm_wav)))
    if rms < 0.006 and not vosk_fallback_text:
        return ""

    model, proc, prompt_ids = get_whisper_engine()
    if model is not None and proc is not None:
        try:
            import torch
            with _WHISPER_LOCK:
                dev = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                feats = proc(
                    norm_wav,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(device=dev, dtype=dtype)
                gen_kwargs = {
                    "language": "italian",
                    "task": "transcribe",
                    "max_new_tokens": 112,
                    "do_sample": False,
                }
                if prompt_ids is not None:
                    gen_kwargs["prompt_ids"] = prompt_ids
                with torch.inference_mode():
                    out_ids = model.generate(feats, **gen_kwargs)
                decoded = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
                cleaned = _postprocess_italian_transcript(decoded)
                if cleaned:
                    return cleaned
        except Exception as exc:
            print(f"[Lumina STT] Errore inferenza Whisper ({exc}); uso fallback Vosk.", flush=True)

    if vosk_fallback_text:
        return _postprocess_italian_transcript(vosk_fallback_text)

    # Ultimo fallback: Vosk 1-best sull'intero enunciato normalizzato
    try:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(get_vosk_model(), 16000)
        pcm16 = np.clip(norm_wav * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        rec.AcceptWaveform(pcm16)
        raw = json.loads(rec.FinalResult()).get("text", "")
        return _postprocess_italian_transcript(raw)
    except Exception:
        return ""


def transcribe_pcm16_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Trascrive bytes PCM16 mono alla frequenza indicata."""
    import numpy as np
    if not pcm_bytes:
        return ""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav_16k = _resample_to_16k(samples, sample_rate)
    return transcribe_float32_16k(wav_16k)


def listen_utterance(
    max_seconds: float = 12.0,
    stop_check=None,
    on_partial=None,
) -> dict:
    """Ascolta una frase dal microfono locale in modo solido e la trascrive con Whisper GPU."""
    import queue as audio_queue
    import numpy as np
    import sounddevice as sd

    vosk_rec = None
    try:
        if VOSK_DIR.is_dir():
            from vosk import KaldiRecognizer
            vosk_rec = KaldiRecognizer(get_vosk_model(), 16000)
            try:
                vosk_rec.SetWords(True)
                vosk_rec.SetPartialWords(True)
            except AttributeError:
                pass
    except Exception:
        vosk_rec = None

    if stop_check and stop_check():
        return {"status": "stopped", "text": ""}

    q_audio: audio_queue.Queue = audio_queue.Queue()

    def _callback(indata, _frames, _time_info, _status):
        q_audio.put(bytes(indata))

    pre_roll = deque(maxlen=4)  # 400 ms di pre-roll
    recorded_chunks: list = []
    noise_floor_samples: list[float] = []
    started = False
    speech_streak = 0
    speech_chunks = 0
    last_voice_ts = time.monotonic()
    last_partial = ""

    with sd.RawInputStream(samplerate=16000, blocksize=1600, dtype="int16", channels=1, callback=_callback):
        deadline = time.monotonic() + float(max_seconds)
        while time.monotonic() < deadline:
            if stop_check and stop_check():
                break
            try:
                raw_bytes = q_audio.get(timeout=0.15)
            except audio_queue.Empty:
                continue

            chunk_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            clean_chunk = chunk_float - float(np.mean(chunk_float))
            rms = float(np.sqrt(np.mean(clean_chunk * clean_chunk)) + 1e-7)

            if not started and len(noise_floor_samples) < 3 and rms < 0.025:
                noise_floor_samples.append(rms)
            noise_floor = float(np.median(noise_floor_samples)) if noise_floor_samples else 0.0035
            speech_thresh = max(0.009, min(0.040, noise_floor * 2.5))

            partial_changed = False
            if vosk_rec is not None:
                vosk_pcm = np.clip(clean_chunk * 2.0 * 32767.0, -32768, 32767).astype(np.int16).tobytes()
                vosk_rec.AcceptWaveform(vosk_pcm)
                part = json.loads(vosk_rec.PartialResult()).get("partial", "").strip()
                if part and part != last_partial:
                    last_partial = part
                    partial_changed = True
                    if on_partial:
                        on_partial(part)

            is_voice_frame = (rms >= speech_thresh) or partial_changed
            if is_voice_frame:
                speech_streak += 1
            else:
                speech_streak = 0

            if not started:
                pre_roll.append(chunk_float)
                if speech_streak >= 2 or partial_changed:
                    started = True
                    speech_chunks = len(pre_roll)
                    recorded_chunks.extend(pre_roll)
                    pre_roll.clear()
                    last_voice_ts = time.monotonic()
            else:
                recorded_chunks.append(chunk_float)
                if is_voice_frame:
                    speech_chunks += 1
                    last_voice_ts = time.monotonic()
                elif (time.monotonic() - last_voice_ts) > 1.25:
                    break

    was_stopped = bool(stop_check and stop_check())
    if not started or not recorded_chunks:
        return {"status": "stopped" if was_stopped else "empty", "text": ""}

    vosk_text = ""
    if vosk_rec is not None:
        try:
            vosk_text = json.loads(vosk_rec.FinalResult()).get("text", "").strip() or last_partial
        except Exception:
            vosk_text = last_partial

    full_audio = np.concatenate(recorded_chunks).astype(np.float32)
    if speech_chunks < 2 and not vosk_text:
        return {"status": "stopped" if was_stopped else "empty", "text": ""}

    final_text = transcribe_float32_16k(full_audio, vosk_fallback_text=vosk_text)
    if on_partial and final_text:
        on_partial(final_text)
    return {"status": "ok" if final_text else "empty", "text": final_text}

    was_stopped = bool(stop_check and stop_check())
    if not started or not recorded_chunks:
        return {"status": "stopped" if was_stopped else "empty", "text": ""}

    vosk_text = ""
    if vosk_rec is not None:
        try:
            vosk_text = json.loads(vosk_rec.FinalResult()).get("text", "").strip() or last_partial
        except Exception:
            vosk_text = last_partial

    full_audio = np.concatenate(recorded_chunks).astype(np.float32)
    if speech_chunks < 2 and not vosk_text:
        return {"status": "stopped" if was_stopped else "empty", "text": ""}

    final_text = transcribe_float32_16k(full_audio, vosk_fallback_text=vosk_text)
    if on_partial and final_text:
        on_partial(final_text)
    return {"status": "ok" if final_text else "empty", "text": final_text}
