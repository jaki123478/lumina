"use strict";
const assert = require("node:assert/strict");
const { fft, resample, extractFeatures, dtwDistance, matchTemplates } = require("../dist/voice-recognizer.js");

const impulse = new Float64Array(512);
const imaginary = new Float64Array(512);
impulse[0] = 1;
fft(impulse, imaginary);
assert(impulse.every(value => Math.abs(value - 1) < 1e-10));
assert(imaginary.every(value => Math.abs(value) < 1e-10));

const sine = Float64Array.from({ length: 512 }, (_, i) => Math.sin(2 * Math.PI * 16 * i / 512));
const sineImaginary = new Float64Array(512);
fft(sine, sineImaginary);
assert(Math.abs(Math.abs(sineImaginary[16]) - 256) < 1e-8);
assert(Math.abs(sine[16]) < 1e-8);
assert.equal(resample(new Float32Array(48000).fill(0.25), 48000).length, 16000);
assert(resample(new Float32Array(48000).fill(0.25), 48000).every(value => Math.abs(value - 0.25) < 1e-6));
assert.throws(() => extractFeatures(new Float32Array(16000)), /bassa/);
assert.throws(() => extractFeatures(new Float32Array(800)), /breve/);
let noiseSeed = 7;
const whiteNoise = Float32Array.from({ length: 16000 }, () => {
  noiseSeed = (1664525 * noiseSeed + 1013904223) >>> 0;
  return (noiseSeed / 4294967296 * 2 - 1) * 0.1;
});
assert.throws(() => extractFeatures(whiteNoise), /rumore/);

function pseudoWord(order, stretch = 1, noise = 0) {
  const sampleRate = 16000;
  const chunkLength = Math.round(sampleRate * 0.19 * stretch);
  const output = new Float32Array(chunkLength * order.length);
  const formants = [[650, 1250, 2400], [300, 2200, 3000], [350, 750, 2200], [450, 1700, 2600]];
  let random = 42;
  for (let segment = 0; segment < order.length; segment += 1) {
    for (let index = 0; index < chunkLength; index += 1) {
      const time = (segment * chunkLength + index) / sampleRate;
      const envelope = Math.sin(Math.PI * index / chunkLength) ** 0.6;
      let value = 0;
      for (let harmonic = 1; harmonic < 35; harmonic += 1) {
        const frequency = harmonic * 130;
        let strength = 0.05 / harmonic;
        for (const formant of formants[order[segment]]) strength += Math.exp(-(((frequency - formant) / 120) ** 2)) / 4;
        value += Math.sin(2 * Math.PI * frequency * time) * strength;
      }
      random = (1664525 * random + 1013904223) >>> 0;
      output[segment * chunkLength + index] = value * envelope * 0.35 + ((random / 4294967296) * 2 - 1) * noise;
    }
  }
  return output;
}

const first = extractFeatures(pseudoWord([0, 1, 2]));
const repeat = extractFeatures(pseudoWord([0, 1, 2], 1.14, 0.002));
const second = extractFeatures(pseudoWord([3, 0, 3]));
assert.equal(first.frames[0].length, 25);
assert(first.frames.every(frame => frame.every(Number.isFinite)));
assert(dtwDistance(first.frames, first.frames) < 1e-12);
const repeatDistance = dtwDistance(first.frames, repeat.frames);
const differentDistance = dtwDistance(first.frames, second.frames);
assert(repeatDistance < differentDistance, `repeat=${repeatDistance}, different=${differentDistance}`);
const templates = [
  { id: "ciao", label: "Ciao", samples: [first, repeat] },
  { id: "aiuto", label: "Aiuto", samples: [second] },
];
assert.equal(matchTemplates(first, templates).text, "Ciao");
assert.equal(matchTemplates(second, templates).text, "Aiuto");
assert.equal(matchTemplates({ frames: first.frames, duration: 20 }, templates), null);

console.log(JSON.stringify({ ok: true, checks: 16, repeatDistance, differentDistance, limitations: "Synthetic DSP tests; real microphone accuracy needs user-recorded samples." }));
