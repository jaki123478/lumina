/* Personalized, limited-vocabulary speech recognition from scratch.
 * Microphone -> local DSP -> MFCC features -> dynamic time warping.
 * All reference recordings are supplied by the user; there are no pretrained
 * weights, speech recognition APIs, cloud requests, or external libraries.
 */
(function (scope) {
  "use strict";

  const STORAGE_KEY = "lumina.voice.templates.v1";
  const SAMPLE_RATE = 16000;
  const FFT_SIZE = 512;
  const FRAME_SIZE = 400;
  const FRAME_STEP = 160;
  const MEL_FILTERS = 26;
  const CEPSTRA = 12;

  function fft(real, imaginary) {
    const size = real.length;
    for (let index = 1, reverse = 0; index < size; index += 1) {
      let bit = size >> 1;
      for (; reverse & bit; bit >>= 1) reverse ^= bit;
      reverse ^= bit;
      if (index < reverse) {
        [real[index], real[reverse]] = [real[reverse], real[index]];
        [imaginary[index], imaginary[reverse]] = [imaginary[reverse], imaginary[index]];
      }
    }
    for (let length = 2; length <= size; length <<= 1) {
      const angle = -2 * Math.PI / length;
      const rootReal = Math.cos(angle);
      const rootImaginary = Math.sin(angle);
      for (let offset = 0; offset < size; offset += length) {
        let rotationReal = 1;
        let rotationImaginary = 0;
        for (let index = 0; index < length / 2; index += 1) {
          const even = offset + index;
          const odd = even + length / 2;
          const oddReal = real[odd] * rotationReal - imaginary[odd] * rotationImaginary;
          const oddImaginary = real[odd] * rotationImaginary + imaginary[odd] * rotationReal;
          real[odd] = real[even] - oddReal;
          imaginary[odd] = imaginary[even] - oddImaginary;
          real[even] += oddReal;
          imaginary[even] += oddImaginary;
          const nextReal = rotationReal * rootReal - rotationImaginary * rootImaginary;
          rotationImaginary = rotationReal * rootImaginary + rotationImaginary * rootReal;
          rotationReal = nextReal;
        }
      }
    }
  }

  function resample(input, sourceRate) {
    if (sourceRate === SAMPLE_RATE) return Float32Array.from(input);
    const ratio = sourceRate / SAMPLE_RATE;
    const output = new Float32Array(Math.floor(input.length / ratio));
    // Integrate input samples over each output interval. This provides a basic
    // antialiasing low-pass for the usual 44.1/48 kHz microphone streams.
    for (let index = 0; index < output.length; index += 1) {
      const start = index * ratio;
      const end = (index + 1) * ratio;
      let value = 0;
      for (let source = Math.floor(start); source < Math.ceil(end); source += 1) {
        const overlap = Math.min(end, source + 1) - Math.max(start, source);
        if (overlap > 0) value += (input[Math.min(source, input.length - 1)] || 0) * overlap;
      }
      output[index] = value / ratio;
    }
    return output;
  }

  function melBank() {
    const low = 2595 * Math.log10(1 + 70 / 700);
    const high = 2595 * Math.log10(1 + 7600 / 700);
    const frequencies = Array.from({ length: MEL_FILTERS + 2 }, (_, index) => {
      const mel = low + (high - low) * index / (MEL_FILTERS + 1);
      return 700 * (10 ** (mel / 2595) - 1);
    });
    return Array.from({ length: MEL_FILTERS }, (_, filter) => {
      const weights = new Float32Array(FFT_SIZE / 2 + 1);
      const [left, middle, right] = frequencies.slice(filter, filter + 3);
      for (let bin = 0; bin < weights.length; bin += 1) {
        const frequency = bin * SAMPLE_RATE / FFT_SIZE;
        weights[bin] = Math.max(0, Math.min((frequency - left) / (middle - left), (right - frequency) / (right - middle)));
      }
      return weights;
    });
  }

  const bank = melBank();
  const hamming = Float32Array.from({ length: FRAME_SIZE }, (_, index) => 0.54 - 0.46 * Math.cos(2 * Math.PI * index / (FRAME_SIZE - 1)));

  function extractFeatures(input, sourceRate = SAMPLE_RATE) {
    if (!input || input.length < sourceRate * 0.18) throw new Error("Registrazione troppo breve. Pronuncia tutta la frase.");
    const signal = resample(input, sourceRate);
    let energy = 0;
    let peak = 0;
    for (const value of signal) {
      energy += value * value;
      peak = Math.max(peak, Math.abs(value));
    }
    const rms = Math.sqrt(energy / signal.length);
    if (rms < 0.003 || peak < 0.015) throw new Error("Voce troppo bassa. Avvicinati al microfono e ripeti.");
    // Trim long silent margins, retaining short context around the word.
    const threshold = Math.max(0.006, peak * 0.025);
    let first = 0;
    let last = signal.length - 1;
    while (first < last && Math.abs(signal[first]) < threshold) first += 1;
    while (last > first && Math.abs(signal[last]) < threshold) last -= 1;
    first = Math.max(0, first - FRAME_SIZE);
    last = Math.min(signal.length, last + FRAME_SIZE);
    const trimmed = signal.subarray(first, last);
    if (trimmed.length < SAMPLE_RATE * 0.16) throw new Error("Non ho rilevato una frase completa.");
    const coefficients = [];
    let averageFlatness = 0;
    for (let start = 0; start + FRAME_SIZE <= trimmed.length; start += FRAME_STEP) {
      const real = new Float64Array(FFT_SIZE);
      const imaginary = new Float64Array(FFT_SIZE);
      let frameEnergy = 0;
      for (let index = 0; index < FRAME_SIZE; index += 1) {
        const sample = trimmed[start + index];
        const previous = start + index > 0 ? trimmed[start + index - 1] : 0;
        real[index] = (sample - 0.97 * previous) * hamming[index];
        frameEnergy += sample * sample;
      }
      fft(real, imaginary);
      const power = new Float64Array(FFT_SIZE / 2 + 1);
      let arithmetic = 0;
      let geometric = 0;
      for (let bin = 1; bin < power.length; bin += 1) {
        power[bin] = real[bin] ** 2 + imaginary[bin] ** 2 + 1e-12;
        // Undo pre-emphasis for the noise diagnostic: otherwise a flat noise
        // spectrum looks sloped and can be mistaken for harmonic speech.
        const emphasis = 1 + 0.97 ** 2 - 1.94 * Math.cos(2 * Math.PI * bin / FFT_SIZE);
        const diagnosticPower = power[bin] / Math.max(1e-5, emphasis);
        arithmetic += diagnosticPower;
        geometric += Math.log(diagnosticPower);
      }
      averageFlatness += Math.exp(geometric / (power.length - 1)) / (arithmetic / (power.length - 1) + 1e-12);
      const logs = bank.map(filter => {
        let sum = 0;
        for (let bin = 1; bin < power.length; bin += 1) sum += power[bin] * filter[bin];
        return Math.log(Math.max(sum, 1e-10));
      });
      const frame = new Float32Array(CEPSTRA + 1);
      for (let coefficient = 1; coefficient <= CEPSTRA; coefficient += 1) {
        let value = 0;
        for (let filter = 0; filter < MEL_FILTERS; filter += 1) {
          value += logs[filter] * Math.cos(Math.PI * coefficient * (filter + 0.5) / MEL_FILTERS);
        }
        frame[coefficient - 1] = value * Math.sqrt(2 / MEL_FILTERS);
      }
      frame[CEPSTRA] = Math.log(frameEnergy / FRAME_SIZE + 1e-10);
      coefficients.push(frame);
    }
    if (coefficients.length < 12) throw new Error("Frase troppo breve per il riconoscimento.");
    averageFlatness /= coefficients.length;
    if (averageFlatness > 0.5) throw new Error("Troppo rumore: registra la frase in un ambiente più silenzioso.");
    // Cepstral mean normalization removes steady microphone coloration.
    const means = new Float64Array(CEPSTRA + 1);
    const deviations = new Float64Array(CEPSTRA + 1);
    for (const frame of coefficients) for (let k = 0; k < frame.length; k += 1) means[k] += frame[k] / coefficients.length;
    for (const frame of coefficients) for (let k = 0; k < frame.length; k += 1) deviations[k] += (frame[k] - means[k]) ** 2 / coefficients.length;
    for (let k = 0; k < deviations.length; k += 1) deviations[k] = Math.max(k === CEPSTRA ? 0.8 : 2, Math.sqrt(deviations[k]));
    for (const frame of coefficients) for (let k = 0; k < frame.length; k += 1) frame[k] = (frame[k] - means[k]) / deviations[k];
    const frames = coefficients.map((frame, index) => {
      const result = Array.from(frame);
      for (let coefficient = 0; coefficient < CEPSTRA; coefficient += 1) {
        let delta = 0;
        for (let offset = 1; offset <= 2; offset += 1) {
          const before = coefficients[Math.max(0, index - offset)][coefficient];
          const after = coefficients[Math.min(coefficients.length - 1, index + offset)][coefficient];
          delta += offset * (after - before) / 10;
        }
        result.push(delta);
      }
      return result.map(value => Math.round(value * 10000) / 10000);
    });
    // Keep storage and comparison bounded for a small, user-trained vocabulary.
    const stride = Math.max(1, Math.ceil(frames.length / 240));
    return { frames: frames.filter((_, index) => index % stride === 0), duration: trimmed.length / SAMPLE_RATE, rms, flatness: averageFlatness };
  }

  function frameDistance(left, right) {
    let sum = 0;
    for (let index = 0; index < left.length; index += 1) sum += (left[index] - right[index]) ** 2;
    return sum / left.length;
  }

  function dtwDistance(left, right) {
    if (!left.length || !right.length || left[0].length !== right[0].length) return Infinity;
    const rows = left.length;
    const columns = right.length;
    // A relative diagonal window tolerates differences in speaking speed.
    const window = Math.max(12, Math.ceil(Math.max(rows, columns) * 0.3));
    let previous = new Float64Array(columns + 1).fill(Infinity);
    let previousLengths = new Uint16Array(columns + 1);
    previous[0] = 0;
    for (let row = 1; row <= rows; row += 1) {
      const current = new Float64Array(columns + 1).fill(Infinity);
      const lengths = new Uint16Array(columns + 1);
      const expected = row * columns / rows;
      const minimum = Math.max(1, Math.floor(expected - window));
      const maximum = Math.min(columns, Math.ceil(expected + window));
      for (let column = minimum; column <= maximum; column += 1) {
        let best = previous[column - 1];
        let length = previousLengths[column - 1];
        if (previous[column] < best) { best = previous[column]; length = previousLengths[column]; }
        if (current[column - 1] < best) { best = current[column - 1]; length = lengths[column - 1]; }
        current[column] = best + frameDistance(left[row - 1], right[column - 1]);
        lengths[column] = length + 1;
      }
      previous = current;
      previousLengths = lengths;
    }
    return previous[columns] / Math.max(1, previousLengths[columns]);
  }

  function matchTemplates(features, templates) {
    const matches = templates.map(template => {
      let distance = Infinity;
      for (const sample of template.samples) {
        const durationRatio = features.duration / sample.duration;
        if (durationRatio < 0.45 || durationRatio > 2.2) continue;
        const candidate = dtwDistance(features.frames, sample.frames) + Math.abs(Math.log(durationRatio)) * 0.12;
        distance = Math.min(distance, candidate);
      }
      let threshold = 0.72;
      if (template.samples.length >= 2) {
        const distances = [];
        for (let a = 0; a < template.samples.length; a += 1) {
          for (let b = a + 1; b < template.samples.length; b += 1) distances.push(dtwDistance(template.samples[a].frames, template.samples[b].frames));
        }
        distances.sort((a, b) => a - b);
        const median = distances[Math.floor(distances.length / 2)];
        threshold = Math.min(0.95, Math.max(0.42, median * 1.65 + 0.12));
      }
      return { id: template.id, text: template.label, distance, threshold };
    }).sort((a, b) => a.distance - b.distance);
    const best = matches[0];
    if (!best || !Number.isFinite(best.distance)) return null;
    const second = matches[1];
    const separated = !second || !Number.isFinite(second.distance) || best.distance < second.distance * 0.88 || second.distance - best.distance > 0.1;
    if (best.distance > best.threshold || !separated) return null;
    return { text: best.text, distance: best.distance, confidence: Math.max(0, Math.min(1, 1 - best.distance / (best.threshold * 1.4))) };
  }

  function joinChunks(chunks) {
    const joined = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) { joined.set(chunk, offset); offset += chunk.length; }
    return joined;
  }

  function validTemplate(template) {
    return template && typeof template.id === "string" && typeof template.label === "string" && template.label.length <= 80
      && Array.isArray(template.samples) && template.samples.length > 0 && template.samples.length <= 5
      && template.samples.every(sample => Number.isFinite(sample.duration) && sample.duration > 0
        && Array.isArray(sample.frames) && sample.frames.length >= 12 && sample.frames.length <= 240
        && sample.frames.every(frame => Array.isArray(frame) && frame.length === 25 && frame.every(Number.isFinite)));
  }

  class LuminaRecognizer {
    constructor(callbacks = {}) {
      this.callbacks = callbacks;
      this.templates = [];
      this._epoch = 0;
      this._mode = "idle";
      try {
        const saved = JSON.parse(scope.localStorage?.getItem(STORAGE_KEY) || "[]");
        if (Array.isArray(saved)) this.templates = saved.filter(validTemplate).slice(0, 20);
      } catch (_) { /* Invalid or unavailable storage starts a fresh vocabulary. */ }
    }

    _status(state, message) { this.callbacks.onStatus?.({ state, message }); }
    _error(message) { this.callbacks.onError?.(message); }
    listTemplates() { return this.templates.map(({ id, label, samples }) => ({ id, label, examples: samples.length })); }
    _save() {
      try { scope.localStorage?.setItem(STORAGE_KEY, JSON.stringify(this.templates)); }
      catch (_) { this._status("idle", "Spazio locale esaurito: i nuovi campioni restano solo in questa sessione."); }
      this.callbacks.onTemplates?.(this.listTemplates());
    }

    deleteTemplate(id) {
      this.templates = this.templates.filter(template => template.id !== id);
      this._save();
    }

    clearTemplates() { this.templates = []; this._save(); }

    async _open(mode) {
      if (!scope.navigator?.mediaDevices?.getUserMedia) throw new Error("Microfono non disponibile. Apri l'app su localhost o tramite HTTPS.");
      const epoch = this._epoch;
      // Chrome can leave getUserMedia pending forever when the permission
      // prompt is blocked or ignored. Never let that freeze the live dialog.
      const permissionTimeout = new Promise((_, reject) => setTimeout(() => reject(new Error("Permesso microfono non ricevuto. Premi l’icona del lucchetto, consenti il microfono e riprova.")), 7000));
      const stream = await Promise.race([
        scope.navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false }),
        permissionTimeout
      ]);
      if (epoch !== this._epoch) {
        stream.getTracks().forEach(track => track.stop());
        throw new Error("Registrazione annullata.");
      }
      this.stream = stream;
      const AudioContextClass = scope.AudioContext || scope.webkitAudioContext;
      if (!AudioContextClass) { this._closeMic(); throw new Error("Web Audio non disponibile in questo browser."); }
      const context = new AudioContextClass();
      this.audioContext = context;
      await context.resume();
      if (epoch !== this._epoch) {
        // A later recording may already own this.audioContext/stream. Release
        // only the resources belonging to this stale asynchronous open.
        stream.getTracks().forEach(track => track.stop());
        if (context.state !== "closed") context.close().catch(() => {});
        throw new Error("Registrazione annullata.");
      }
      this.source = this.audioContext.createMediaStreamSource(stream);
      this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
      this.silencer = this.audioContext.createGain();
      this.silencer.gain.value = 0;
      this.source.connect(this.processor);
      this.processor.connect(this.silencer);
      this.silencer.connect(this.audioContext.destination);
      this._sampleRate = this.audioContext.sampleRate;
      this._mode = mode;
      this._noiseFloor = 0.003;
      this._clock = 0;
      this._busy = false;
      this._resetCapture();
      this.processor.onaudioprocess = event => this._audio(Float32Array.from(event.inputBuffer.getChannelData(0)));
      this._status(mode === "record" ? "recording" : "listening", mode === "record" ? "Pronuncia la frase, poi fai una pausa." : "In ascolto delle frasi che hai registrato.");
    }

    _resetCapture() {
      this._preRoll = [];
      this._chunks = [];
      this._capturing = false;
      this._voicedFrames = 0;
      this._silenceDuration = 0;
      this._captureDuration = 0;
    }

    _audio(samples) {
      if (this._mode === "idle" || this._busy) return;
      const duration = samples.length / this._sampleRate;
      this._clock += duration;
      let energy = 0;
      let crossings = 0;
      for (let index = 0; index < samples.length; index += 1) {
        energy += samples[index] ** 2;
        if (index && (samples[index] >= 0) !== (samples[index - 1] >= 0)) crossings += 1;
      }
      const rms = Math.sqrt(energy / samples.length);
      const voice = rms > Math.max(0.009, this._noiseFloor * 3) && crossings / samples.length < 0.4;
      if (!this._capturing) {
        this._preRoll.push(samples);
        while (this._preRoll.length * duration > 0.25) this._preRoll.shift();
        if (!voice && rms < 0.025) this._noiseFloor = this._noiseFloor * 0.97 + rms * 0.03;
        this._voicedFrames = voice ? this._voicedFrames + 1 : 0;
        if (this._voicedFrames >= 2 && this._clock > 0.15) {
          this._capturing = true;
          this._chunks = this._preRoll.slice();
          this._captureDuration = this._chunks.length * duration;
          this._silenceDuration = 0;
        }
        return;
      }
      this._chunks.push(samples);
      this._captureDuration += duration;
      this._silenceDuration = voice ? 0 : this._silenceDuration + duration;
      if ((this._silenceDuration > 0.62 && this._captureDuration > 0.4) || this._captureDuration > 4.8) {
        const chunks = this._chunks;
        const mode = this._mode;
        this._resetCapture();
        this._finish(joinChunks(chunks), mode);
      }
    }

    _finish(samples, mode) {
      this._busy = true;
      this._status("processing", "Confronto della voce con i tuoi campioni…");
      const rate = this._sampleRate;
      if (mode === "record") this._closeMic();
      try {
        const features = extractFeatures(samples, rate);
        if (mode === "record") {
          const pending = this._pending;
          this._pending = undefined;
          clearTimeout(this._recordTimeout);
          let template = this.templates.find(item => item.label.toLocaleLowerCase("it") === pending.label.toLocaleLowerCase("it"));
          if (!template) {
            template = { id: scope.crypto?.randomUUID?.() || `phrase-${Date.now()}-${Math.random().toString(36).slice(2)}`, label: pending.label, samples: [] };
            this.templates.push(template);
          }
          template.samples.push(features);
          if (template.samples.length > 5) template.samples.shift();
          this._save();
          this._status("idle", "Campione salvato sul dispositivo.");
          pending.resolve({ id: template.id, label: template.label, examples: template.samples.length });
        } else {
          const result = matchTemplates(features, this.templates);
          if (result) this.callbacks.onResult?.(result);
          if (this._mode === "live") this._status("listening", result ? "Frase riconosciuta." : "Frase non riconosciuta. Ripeti una delle frasi registrate.");
        }
      } catch (error) {
        if (mode === "record") {
          const pending = this._pending;
          this._pending = undefined;
          clearTimeout(this._recordTimeout);
          pending?.reject(error);
          this._error(error.message);
        } else if (this._mode === "live") this._status("listening", error.message);
      } finally { this._busy = false; }
    }

    async record(label) {
      label = String(label || "").trim();
      if (!label || label.length > 80) throw new Error("Scrivi una frase tra 1 e 80 caratteri da associare alla voce.");
      if (this.templates.length >= 20 && !this.templates.some(template => template.label.toLocaleLowerCase("it") === label.toLocaleLowerCase("it"))) throw new Error("Puoi registrare fino a 20 frasi. Elimina una frase per aggiungerne un'altra.");
      this.stop();
      const epoch = this._epoch;
      try { await this._open("record"); }
      catch (error) {
        if (epoch === this._epoch) { this._closeMic(); this._error(error.message); }
        throw error;
      }
      return new Promise((resolve, reject) => {
        this._pending = { label, resolve, reject };
        this._recordTimeout = setTimeout(() => {
          this._pending = undefined;
          this._closeMic();
          const error = new Error("Nessuna frase rilevata. Controlla il microfono e registra di nuovo.");
          this._error(error.message);
          reject(error);
        }, 12000);
      });
    }

    async start() {
      if (!this.templates.length) throw new Error("Registra prima almeno una frase nella sezione Addestra voce.");
      this.stop();
      const epoch = this._epoch;
      try { await this._open("live"); }
      catch (error) {
        if (epoch === this._epoch) { this._closeMic(); this._error(error.message); }
        throw error;
      }
    }

    _closeMic() {
      this._mode = "idle";
      if (this.processor) { this.processor.onaudioprocess = null; this.processor.disconnect(); }
      this.source?.disconnect();
      this.silencer?.disconnect();
      this.stream?.getTracks().forEach(track => track.stop());
      if (this.audioContext && this.audioContext.state !== "closed") this.audioContext.close().catch(() => {});
      this.processor = this.source = this.silencer = this.stream = this.audioContext = undefined;
    }

    stop() {
      this._epoch += 1;
      clearTimeout(this._recordTimeout);
      const pending = this._pending;
      this._pending = undefined;
      this._closeMic();
      pending?.reject(new Error("Registrazione annullata."));
      this._status("idle", "Microfono spento.");
    }

    destroy() { this.stop(); this.callbacks = {}; }
  }

  scope.LuminaRecognizer = LuminaRecognizer;
  if (typeof module !== "undefined" && module.exports) module.exports = { LuminaRecognizer, extractFeatures, dtwDistance, matchTemplates, fft, resample };
})(typeof window !== "undefined" ? window : globalThis);
