"use strict";
const fs = require("node:fs");
const { LuminaEngine } = require("../dist/inference-worker.js");
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const engine = new LuminaEngine(fixture.artifact);
const assert = require("node:assert/strict");
for (const prompt of fixture.prompts) assert.deepEqual(engine.prompt(prompt.messages).tokens, prompt.tokens, "browser and PyTorch prompts differ");
let maximumDifference = 0;
for (const example of fixture.examples) {
  const actual = engine.prefill(example.tokens);
  for (let index = 0; index < actual.length; index += 1) maximumDifference = Math.max(maximumDifference, Math.abs(actual[index] - example.expected[index]));
}
const rollover = fixture.rollover;
engine.prefill(rollover.tokens);
let actual;
for (const token of rollover.append) actual = engine.append(token);
for (let index = 0; index < actual.length; index += 1) maximumDifference = Math.max(maximumDifference, Math.abs(actual[index] - rollover.expected[index]));
if (maximumDifference > 0.00015) throw new Error(`JavaScript/PyTorch logits differ by ${maximumDifference}`);
console.log(JSON.stringify({ parameters: engine.parameterCount, maximumDifference, examples: fixture.examples.length, rollover: true }));
