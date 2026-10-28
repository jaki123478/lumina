// Piper viene eseguito dal server su questo computer. Nessun servizio remoto.
export class LocalVoiceSynth {
  constructor() { this.epoch = 0; this.audio = null; this.controller = null; this.finish = null; }
  async unlock() {}
  stop() {
    this.epoch += 1;
    this.controller?.abort(); this.controller = null;
    if (this.audio) { this.audio.pause(); this.audio = null; }
    this.finish?.(); this.finish = null;
  }
  async speak(text) {
    this.stop(); const epoch = this.epoch;
    const controller = new AbortController(); this.controller = controller;
    let url;
    try {
      const response = await fetch('/api/tts', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal: controller.signal });
      if (!response.ok) throw new Error('Voce locale non disponibile. Controlla il server Lumina.');
      const blob = await response.blob();
      if (epoch !== this.epoch) return;
      url = URL.createObjectURL(blob);
      const audio = new Audio(url); this.audio = audio;
      await new Promise((resolve, reject) => {
        this.finish = resolve;
        audio.onended = resolve; audio.onerror = () => reject(new Error('Riproduzione audio non riuscita.'));
        audio.play().catch(reject);
      });
    } catch (error) {
      if (epoch === this.epoch && error.name !== 'AbortError') throw error;
    } finally {
      if (url) URL.revokeObjectURL(url);
      if (epoch === this.epoch) { this.audio = null; this.finish = null; this.controller = null; }
    }
  }
}
