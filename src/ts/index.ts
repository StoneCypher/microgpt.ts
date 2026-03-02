/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm. Everything else is just efficiency.
 *
 * Converted from Karpathy's microgpt.py.
 *
 * @module microgpt
 */

import type { TrainType } from './types';
export type { TrainType };
export type { Matrix, StateDict, GPTConfig, AdamConfig, AdamState } from './types';

import { createStateDict } from './gpt.js';
import { generate } from './inference.js';
import { SeededRandom } from './rng.js';
import { buildTokenizer, tokenize } from './tokenizer.js';
import { createAdamState, trainStep } from './training.js';

export { SeededRandom };
export { generate };
export { createAdamState, trainStep };
export { buildTokenizer, tokenize };

export { Value } from './autograd.js';
export { linear, softmax, rms_norm } from './nn_helpers.js';
export { createStateDict, gpt } from './gpt.js';





export async function train({
  seed = 42, docs,
  nLayer = 1, nEmbd = 16, blockSize = 16, nHead = 4,
  learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, eps = 1e-8, temperature = 0.5,
  numSteps = 1000, commentNthStep = 100, sampleCount = 10,
  onStep, signal,
}: TrainType) {

  if (docs === undefined)     { throw new Error('Must define docs'); }
  if (!(Array.isArray(docs))) { throw new Error('Docs must be an array of strings'); }

  // 1. Prepare data
  const rng = new SeededRandom(seed);
  rng.shuffle(docs);

  // 2. Build tokenizer
  const { uchars, BOS, vocabSize } = buildTokenizer(docs);

  // 3. Initialize model
  const config = { nLayer, nEmbd, blockSize, nHead },
        { stateDict, params } = createStateDict(vocabSize, config, rng);

  // 4. Train
  const adamCfg = { learningRate, beta1, beta2, eps },
        adamState = createAdamState(params.length);

  for (let step = 0; step < numSteps; step++) {
    if (signal?.aborted) break;
    const doc = docs[step % docs.length],
          tokens = tokenize(doc, uchars, BOS),
          loss = trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, numSteps);

    const isComment = (step % commentNthStep) === 0;

    if (isComment) console.log(`step ${step} | loss ${loss.toFixed(4)}`);

    if (onStep) {
      const samples: string[] = [];
      if (isComment)
        for (let i = 0; i < sampleCount; i++)
          samples.push(generate(stateDict, config, uchars, BOS, rng, temperature));
      onStep({ step, numSteps, loss, samples });
    }

    // yield to the event loop so the browser can repaint
    if (isComment) await new Promise(r => setTimeout(r, 0));
  }

  // 5. Generate
  const finalSamples: string[] = [];
  for (let i = 0; i < sampleCount; i++) {
    const sample = generate(stateDict, config, uchars, BOS, rng, temperature);
    finalSamples.push(sample);
    console.log(sample);
  }

  return finalSamples;
}
