"""
lumina_voice_neural.py - Voce femminile neurale italiana (Piper VITS "Paola") per Lumina.
100% offline dopo il download del modello. Uso:
  python lumina_voice_neural.py "testo da dire" [--out file.wav] [--no-play]
"""
import argparse
import sys
import wave
import re
import io
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_CANDIDATES = [
    HERE / "voices" / "it_IT-paola-medium.onnx",
]

_voice = None
_voice_lock = threading.RLock()


def load_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        model = next((p for p in VOICE_CANDIDATES if p.exists()), None)
        if model is None:
            raise FileNotFoundError("Modello it_IT-paola-medium.onnx non trovato")
        _voice = PiperVoice.load(str(model))
    return _voice


def clean(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    for marker in ('<|im_end|>', '<|im_start|>', '<|pad|>'):
        text = text.split(marker, 1)[0]
    text = re.sub(r'<think>.*', '', text, flags=re.S)
    text = re.sub(r'```.*?```', ' Il codice è disponibile a schermo. ', text, flags=re.S)
    text = re.sub(r'```.*', ' Il codice è disponibile a schermo. ', text, flags=re.S)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(?m)^\s*(?:#{1,6}|[-*])\s+', '', text)
    text = text.replace('**', '').replace('`', '').replace('km²', 'chilometri quadrati')
    text = text.replace('×', ' per ').replace('CO2', 'anidride carbonica')
    text = ' '.join(text.split())
    if len(text) > 1000:
        text = text[:1000].rsplit(' ', 1)[0] + '.'
    return text


def synthesize(text: str, out_path: Path, *, length_scale: float = 1.08, volume: float = 0.85) -> Path:
    out_path = Path(out_path)
    audio = synthesize_bytes(text, length_scale=length_scale, volume=volume)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    return out_path


def synthesize_bytes(text: str, *, length_scale: float = 1.08, volume: float = 0.85) -> bytes:
    from piper import SynthesisConfig
    text = clean(text)
    if not text:
        raise ValueError('Nessun testo pronunciabile')
    audio = io.BytesIO()
    with _voice_lock:
        voice = load_voice()
        with wave.open(audio, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(
                length_scale=length_scale, volume=volume, normalize_audio=True))
    return audio.getvalue()


def play(path: Path):
    import winsound
    winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME)


def speak(text: str, name: str = "lumina_neurale", do_play: bool = True) -> Path:
    out = HERE / "audio_neurale" / f"{name}.wav"
    synthesize(text, out)
    if do_play:
        play(out)
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args()
    p = Path(a.out) if a.out else HERE / "audio_neurale" / "lumina_neurale.wav"
    synthesize(a.text, p)
    print("WAV:", p)
    if not a.no_play:
        play(p)
