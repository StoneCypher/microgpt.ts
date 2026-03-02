/**
 * Shared type definitions for MicroGPT.
 *
 * @module types
 */

import type { Value } from './autograd';

/** A 2-D array of {@link Value} objects representing a weight matrix. */
export type Matrix = Value[][];

/** String-keyed collection of weight matrices. */
export type StateDict = Record<string, Matrix>;

/**
 * Hyperparameters for the GPT model.
 *
 * @property nLayer    - Number of transformer layers (depth).
 * @property nEmbd     - Embedding dimension (width).
 * @property blockSize - Maximum context length of the attention window.
 * @property nHead     - Number of attention heads.
 */
export interface GPTConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
}

/**
 * Configuration for the Adam optimizer.
 *
 * @property learningRate - Base learning rate.
 * @property beta1        - Exponential decay rate for first moment.
 * @property beta2        - Exponential decay rate for second moment.
 * @property eps          - Small constant for numerical stability.
 */
export interface AdamConfig {
  learningRate: number;
  beta1: number;
  beta2: number;
  eps: number;
}

/** Mutable Adam optimizer state (first and second moment buffers). */
export interface AdamState {
  m: number[];
  v: number[];
}

export type TrainType = {
  seed: number, docs: string[],
  nLayer: number, nEmbd: number, blockSize: number, nHead: number,
  learningRate: number, beta1: number, beta2: number, eps: number, temperature: number,
  numSteps: number, commentNthStep: number, sampleCount: number,
  onStep?: (update: { step: number; numSteps: number; loss: number; samples: string[] }) => void,
  signal?: { aborted: boolean }
};
