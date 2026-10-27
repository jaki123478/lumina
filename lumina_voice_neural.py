"""
lumina_voice_neural.py - Voce femminile neurale italiana (Piper VITS "Paola") per Lumina.
100% offline dopo il download del modello. Uso:
  python lumina_voice_neural.py "testo da dire" [--out file.wav] [--no-play]
"""
import argparse
import sys
import wave
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_CANDIDATES = [
    HERE / "voices" / "it_IT-paola-medium.onnx",
    HERE.parent / "voices" / "it_IT-paola-medium.onnx",
    Path(r"C:\Users\Jaki1\Desktop\ai lumina\voices\it_IT-paola-medium.onnx"),
    Path(r"C:\Users\Jaki1\Desktop\test\voce_neurale_lumina\voices\it_IT-paola-medium.onnx"),
]

_voice = None


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
    for t in ("<|im_start|>", "<|im_end|>", "<|pad|>", "<think>", "</think>"):
        text = text.replace(t, "")
    return " ".join(text.split())[:600]


def synthesize(text: str, out_path: Path) -> Path:
    voice = load_voice()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        voice.synthesize_wav(clean(text), wf)
    return out_path


def play(path: Path):
    ps = (
        "Add-Type -AssemblyName PresentationCore;"
        "$p=New-Object System.Windows.Media.MediaPlayer;"
        f"$p.Open([Uri]'{path}');$p.Volume=1;$p.Play();"
        "Start-Sleep -Milliseconds 800;"
        "while($p.NaturalDuration.HasTimeSpan -and $p.Position -lt $p.NaturalDuration.TimeSpan){Start-Sleep -Milliseconds 300};"
        "$p.Close()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


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
