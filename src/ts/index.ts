/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm. Everything else is just efficiency.
 *
 * Converted from Karpathy's microgpt.py.
 *
 * @module microgpt
 */

export type { Matrix, StateDict, GPTConfig, AdamConfig, AdamState } from './types';

export { Value } from './autograd.js';
export { SeededRandom } from './rng.js';
export { linear, softmax, rms_norm } from './nn_helpers.js';
export { createStateDict, gpt } from './gpt.js';
export { generate } from './inference.js';
export { createAdamState, trainStep } from './training.js';
export { buildTokenizer, tokenize } from './tokenizer.js';
