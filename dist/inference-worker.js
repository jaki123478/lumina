/* Lumina 50K: causal Transformer inference, written for this model.
 * No remote inference service or pretrained model is used.
 * Matrices are exported in PyTorch's [out_features, in_features] order.
 */
"use strict";

const GELU_COEFFICIENT = Math.sqrt(2 / Math.PI);
const MEMORY_BUDGET = 84; // bytes of prompt (incl. <bos>) allowed when remembering the last exchange

function tensor(values, rows, columns, name) {
  const expected = columns === undefined ? rows : rows * columns;
  if (!Array.isArray(values)) throw new Error(`Peso mancante: ${name}`);
  const flat = columns === undefined ? values : values.flat();
  if (flat.length !== expected || flat.some(value => !Number.isFinite(value))) {
    throw new Error(`Dimensioni o valori non validi per ${name}`);
  }
  return new Float32Array(flat);
}

function linear(input, weight, bias, outputSize) {
  const inputSize = input.length;
  const output = new Float32Array(outputSize);
  for (let row = 0; row < outputSize; row += 1) {
    let sum = bias ? bias[row] : 0;
    const offset = row * inputSize;
    for (let column = 0; column < inputSize; column += 1) {
      sum += weight[offset + column] * input[column];
    }
    output[row] = sum;
  }
  return output;
}

function layerNorm(input, weight, bias, epsilon) {
  let mean = 0;
  for (const value of input) mean += value;
  mean /= input.length;
  let variance = 0;
  for (const value of input) variance += (value - mean) ** 2;
  const inverse = 1 / Math.sqrt(variance / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = (input[index] - mean) * inverse * weight[index] + bias[index];
  }
  return output;
}

function residual(left, right) {
  const output = new Float32Array(left.length);
  for (let index = 0; index < left.length; index += 1) output[index] = left[index] + right[index];
  return output;
}

class LuminaEngine {
  constructor(artifact) {
    if (!artifact || !artifact.config || !artifact.state_dict) throw new Error("File del modello non valido.");
    this.config = artifact.config;
    this.training = artifact.training || {};
    const c = this.config;
    for (const key of ["d_model", "n_layers", "n_heads", "ff_dim", "context_length", "vocab_size"]) {
      if (!Number.isInteger(c[key]) || c[key] <= 0) throw new Error(`Configurazione non valida: ${key}`);
    }
    if (c.d_model % c.n_heads || c.vocab_size !== 259 || c.d_model > 256 || c.n_layers > 8 || c.context_length > 2048) {
      throw new Error("Architettura del modello non supportata.");
    }
    this.width = c.d_model;
    this.headWidth = c.d_model / c.n_heads;
    this.epsilon = c.layer_norm_eps || 1e-5;
    this.weights = {};
    let parameterCount = 0;
    const read = (name, rows, columns) => {
      const value = tensor(artifact.state_dict[name], rows, columns, name);
      parameterCount += value.length;
      this.weights[name] = value;
      return value;
    };
    this.embedding = read("token_embedding.weight", c.vocab_size, c.d_model);
    this.positions = read("position_embedding.weight", c.context_length, c.d_model);
    this.blocks = [];
    for (let layer = 0; layer < c.n_layers; layer += 1) {
      const prefix = `blocks.${layer}.`;
      this.blocks.push({
        ln1Weight: read(`${prefix}ln1.weight`, c.d_model),
        ln1Bias: read(`${prefix}ln1.bias`, c.d_model),
        qkvWeight: read(`${prefix}attn.qkv.weight`, c.d_model * 3, c.d_model),
        qkvBias: read(`${prefix}attn.qkv.bias`, c.d_model * 3),
        projectionWeight: read(`${prefix}attn.proj.weight`, c.d_model, c.d_model),
        projectionBias: read(`${prefix}attn.proj.bias`, c.d_model),
        ln2Weight: read(`${prefix}ln2.weight`, c.d_model),
        ln2Bias: read(`${prefix}ln2.bias`, c.d_model),
        fc1Weight: read(`${prefix}fc1.weight`, c.ff_dim, c.d_model),
        fc1Bias: read(`${prefix}fc1.bias`, c.ff_dim),
        fc2Weight: read(`${prefix}fc2.weight`, c.d_model, c.ff_dim),
        fc2Bias: read(`${prefix}fc2.bias`, c.d_model),
        keys: new Float32Array(c.context_length * c.d_model),
        values: new Float32Array(c.context_length * c.d_model),
      });
    }
    this.finalWeight = read("final_norm.weight", c.d_model);
    this.finalBias = read("final_norm.bias", c.d_model);
    if (c.parameter_count !== parameterCount) {
      throw new Error(`Il modello contiene ${parameterCount} parametri: ne sono dichiarati ${c.parameter_count}.`);
    }
    this.parameterCount = parameterCount;
    this.reset();
  }

  reset() {
    this.cachedLength = 0;
    this.sequence = [];
  }

  forwardToken(tokenId, position = this.cachedLength) {
    const c = this.config;
    const d = this.width;
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= c.vocab_size) throw new Error("Token non valido.");
    if (position < 0 || position >= c.context_length || position > this.cachedLength) throw new Error("Posizione fuori contesto.");
    let hidden = new Float32Array(d);
    for (let index = 0; index < d; index += 1) {
      hidden[index] = this.embedding[tokenId * d + index] + this.positions[position * d + index];
    }
    for (const block of this.blocks) {
      const normalized = layerNorm(hidden, block.ln1Weight, block.ln1Bias, this.epsilon);
      const qkv = linear(normalized, block.qkvWeight, block.qkvBias, d * 3);
      block.keys.set(qkv.subarray(d, d * 2), position * d);
      block.values.set(qkv.subarray(d * 2, d * 3), position * d);
      const attention = new Float32Array(d);
      const scores = new Float64Array(position + 1);
      const scale = 1 / Math.sqrt(this.headWidth);
      for (let head = 0; head < c.n_heads; head += 1) {
        const headOffset = head * this.headWidth;
        let maximum = -Infinity;
        for (let previous = 0; previous <= position; previous += 1) {
          let score = 0;
          const keyOffset = previous * d + headOffset;
          for (let index = 0; index < this.headWidth; index += 1) {
            score += qkv[headOffset + index] * block.keys[keyOffset + index];
          }
          scores[previous] = score * scale;
          maximum = Math.max(maximum, scores[previous]);
        }
        let denominator = 0;
        for (let previous = 0; previous <= position; previous += 1) {
          scores[previous] = Math.exp(scores[previous] - maximum);
          denominator += scores[previous];
        }
        for (let index = 0; index < this.headWidth; index += 1) {
          let sum = 0;
          for (let previous = 0; previous <= position; previous += 1) {
            sum += scores[previous] / denominator * block.values[previous * d + headOffset + index];
          }
          attention[headOffset + index] = sum;
        }
      }
      hidden = residual(hidden, linear(attention, block.projectionWeight, block.projectionBias, d));
      const feedforwardInput = layerNorm(hidden, block.ln2Weight, block.ln2Bias, this.epsilon);
      const feedforward = linear(feedforwardInput, block.fc1Weight, block.fc1Bias, c.ff_dim);
      for (let index = 0; index < feedforward.length; index += 1) {
        const value = feedforward[index];
        feedforward[index] = 0.5 * value * (1 + Math.tanh(GELU_COEFFICIENT * (value + 0.044715 * value ** 3)));
      }
      hidden = residual(hidden, linear(feedforward, block.fc2Weight, block.fc2Bias, d));
    }
    this.cachedLength = position + 1;
    const normalized = layerNorm(hidden, this.finalWeight, this.finalBias, this.epsilon);
    return linear(normalized, this.embedding, null, c.vocab_size);
  }

  prefill(tokens) {
    this.reset();
    this.sequence = Array.from(tokens).slice(-this.config.context_length);
    if (!this.sequence.length) this.sequence = [this.config.bos_id];
    let logits;
    for (let position = 0; position < this.sequence.length; position += 1) {
      logits = this.forwardToken(this.sequence[position], position);
    }
    return logits;
  }

  append(tokenId) {
    if (this.sequence.length >= this.config.context_length) {
      return this.prefill([...this.sequence.slice(1), tokenId]);
    }
    this.sequence.push(tokenId);
    return this.forwardToken(tokenId);
  }

  prompt(messages) {
    let userIndex = -1;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index] && messages[index].role === "user") { userIndex = index; break; }
    }
    const user = userIndex >= 0 ? messages[userIndex] : null;
    if (!user || typeof user.content !== "string" || !user.content.trim()) throw new Error("Scrivi un messaggio prima di inviare.");
    const encoder = new TextEncoder();
    const prefix = [...encoder.encode("U: ")];
    const suffix = [...encoder.encode("\nL:")];
    // A small context leaves room for an answer. Only the most recent completed exchange is
    // remembered, and only when the whole prompt fits MEMORY_BUDGET (mirrors prompt_format.py).
    const limit = Math.min(this.config.context_length - prefix.length - suffix.length - 1, 112);
    const characters = Array.from(user.content.trim());
    let bytes = [...encoder.encode(characters.join(""))];
    while (bytes.length > limit) {
      characters.pop();
      bytes = [...encoder.encode(characters.join(""))];
    }
    const tokens = [this.config.bos_id, ...prefix, ...bytes, ...suffix];
    const clean = text => text.split(/\s+/).filter(Boolean).join(" ");
    const answer = userIndex >= 2 ? messages[userIndex - 1] : null;
    const question = userIndex >= 2 ? messages[userIndex - 2] : null;
    if (answer && question && answer.role === "assistant" && question.role === "user"
        && typeof answer.content === "string" && typeof question.content === "string") {
      const previousQuestion = clean(question.content), previousAnswer = clean(answer.content);
      if (previousQuestion && previousAnswer) {
        const history = [...encoder.encode(`U: ${previousQuestion}\nL: ${previousAnswer}\n`)];
        if (history.length + tokens.length <= MEMORY_BUDGET) tokens.splice(1, 0, ...history);
      }
    }
    return { tokens, truncated: characters.join("") !== user.content.trim() };
  }
}

function sample(logits, temperature, utf8, generated, config) {
  // UTF-8 validity is a decoding constraint, not a source of canned answers.
  const candidates = [];
  for (let token = 0; token < logits.length; token += 1) {
    let allowed;
    if (utf8.remaining) {
      allowed = token >= utf8.minimum && token <= utf8.maximum;
    } else {
      allowed = token === config.eos_id || token === 10 || token === 9 || (token >= 32 && token <= 126) || (token >= 194 && token <= 244);
    }
    if (!allowed || !Number.isFinite(logits[token])) continue;
    let score = logits[token];
    if (generated.length > 16 && generated.slice(-20).filter(value => value === token).length >= 8) score -= 0.8;
    candidates.push({ token, score });
  }
  candidates.sort((left, right) => right.score - left.score);
  if (!candidates.length) throw new Error("Il modello ha prodotto probabilità non valide.");
  if (temperature <= 0.001) return candidates[0].token;
  const top = candidates.slice(0, 32);
  const maximum = top[0].score;
  let total = 0;
  for (const candidate of top) {
    candidate.probability = Math.exp((candidate.score - maximum) / temperature);
    total += candidate.probability;
  }
  let choice = Math.random() * total;
  for (const candidate of top) {
    choice -= candidate.probability;
    if (choice <= 0) return candidate.token;
  }
  return top[top.length - 1].token;
}

function updateUtf8(state, token) {
  if (state.remaining) {
    state.remaining -= 1;
    state.minimum = 128;
    state.maximum = 191;
  } else if (token >= 194 && token <= 223) {
    state.remaining = 1;
    state.minimum = 128;
    state.maximum = 191;
  } else if (token >= 224 && token <= 239) {
    state.remaining = 2;
    state.minimum = token === 224 ? 160 : 128;
    state.maximum = token === 237 ? 159 : 191;
  } else if (token >= 240 && token <= 244) {
    state.remaining = 3;
    state.minimum = token === 240 ? 144 : 128;
    state.maximum = token === 244 ? 143 : 191;
  }
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  let engine;
  let loading;
  let active;
  const send = message => self.postMessage(message);
  const pause = () => new Promise(resolve => setTimeout(resolve, 0));

  async function load() {
    if (engine) return engine;
    if (loading) return loading;
    loading = (async () => {
      send({ type: "progress", progress: 0 });
      const response = await fetch("./model/weights.json", { cache: "no-cache" });
      if (!response.ok) throw new Error(`Modello non disponibile (HTTP ${response.status}). Esegui prima l'addestramento.`);
      const artifact = await response.json();
      engine = new LuminaEngine(artifact);
      send({ type: "progress", progress: 1 });
      send({ type: "ready", model: { ...engine.config, parameters: engine.parameterCount, training: engine.training } });
      return engine;
    })();
    try {
      return await loading;
    } catch (error) {
      loading = undefined;
      throw error;
    }
  }

  async function generate(message) {
    if (active) active.cancelled = true;
    const run = { id: message.id, cancelled: false };
    active = run;
    const model = await load();
    if (run.cancelled || active !== run) return;
    const prompt = model.prompt(Array.isArray(message.messages) ? message.messages : []);
    send({ type: "started", id: run.id, truncated: prompt.truncated });
    let logits = model.prefill(prompt.tokens);
    const temperature = Number.isFinite(message.temperature) ? Math.max(0, Math.min(1.5, message.temperature)) : 0.25;
    const maximum = Number.isFinite(message.maxTokens) ? Math.max(1, Math.min(512, Math.round(message.maxTokens))) : 150;
    const generated = [];
    const decoder = new TextDecoder("utf-8", { fatal: false });
    const utf8 = { remaining: 0, minimum: 128, maximum: 191 };
    const start = performance.now();
    let text = "";
    let reason = "length";
    for (let count = 0; count < maximum || (utf8.remaining && count < maximum + 3); count += 1) {
      if (run.cancelled || active !== run) {
        reason = "cancelled";
        break;
      }
      const token = sample(logits, temperature, utf8, generated, model.config);
      if (token === model.config.eos_id) {
        reason = "eos";
        break;
      }
      generated.push(token);
      updateUtf8(utf8, token);
      const chunk = decoder.decode(new Uint8Array([token]), { stream: true });
      if (chunk) {
        text += chunk;
        send({ type: "token", id: run.id, text: chunk });
      }
      // Yield so stop requests can interrupt computation in a Web Worker.
      if (count % 3 === 0) await pause();
      if (run.cancelled || active !== run) {
        reason = "cancelled";
        break;
      }
      logits = model.append(token);
    }
    // If cancellation interrupted a multibyte sequence, omit its incomplete tail.
    if (!utf8.remaining) text += decoder.decode();
    send({ type: "done", id: run.id, text: text.trim(), tokens: generated.length,
      durationMs: Math.round(performance.now() - start), reason, truncated: prompt.truncated });
    if (active === run) active = undefined;
  }

  self.onmessage = event => {
    const message = event.data || {};
    if (message.type === "cancel") {
      if (active && (message.id === undefined || active.id === message.id)) active.cancelled = true;
      return;
    }
    const operation = message.type === "load" ? load() : message.type === "generate" ? generate(message) : null;
    if (operation) operation.catch(error => {
      if (active && active.id === message.id) active = undefined;
      send({ type: "error", id: message.id, message: error.message || "Errore durante l'esecuzione del modello." });
    });
  };
}

if (typeof module !== "undefined" && module.exports) module.exports = { LuminaEngine, sample, updateUtf8 };
