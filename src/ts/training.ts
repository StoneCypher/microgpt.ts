/**
 * Training logic for MicroGPT.
 *
 * @module training
 */

import { Value } from './autograd';
import { softmax } from './nn_helpers';
import { gpt } from './gpt';
import type { StateDict, GPTConfig, AdamConfig, AdamState } from './types';





/** Create a zeroed Adam state for `n` parameters. */
export function createAdamState(n: number): AdamState {
  return { m: new Array<number>(n).fill(0), v: new Array<number>(n).fill(0) };
}



/**
 * Perform a single training step: forward pass, loss computation, backward pass,
 * and Adam parameter update.
 *
 * @param tokens     Token id sequence for one document (including BOS sentinels).
 * @param params     Flat array of all model parameters.
 * @param stateDict  Model weights.
 * @param config     Model hyperparameters.
 * @param adamCfg    Adam optimizer hyperparameters.
 * @param adamState  Mutable Adam moment buffers.
 * @param step       Current training step (0-indexed), used for bias correction and LR decay.
 * @param numSteps   Total number of training steps, used for linear LR decay.
 * @returns  The scalar loss value for this step.
 */

export function trainStep(
  tokens: number[],
  params: Value[],
  stateDict: StateDict,
  config: GPTConfig,
  adamCfg: AdamConfig,
  adamState: AdamState,
  step: number,
  numSteps: number,
): number {

  const n = Math.min(config.blockSize, tokens.length - 1),
        keys: Value[][][] = Array.from({ length: config.nLayer }, () => []),
        valuesCache: Value[][][] = Array.from({ length: config.nLayer }, () => []);

  const losses: Value[] = [];
  for (let posId = 0; posId < n; posId++) {

    const tokenId = tokens[posId]!,
          targetId = tokens[posId + 1]!,
          logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config),
          probs = softmax(logits);

          losses.push(probs[targetId]!.log().neg());

  }

  let loss = losses[0]!;

  for (let i = 1; i < losses.length; i++) {
    loss = loss.add(losses[i]!);
  }

  loss = loss.div(n);

  // Backward
  loss.backward();

  // Adam update
  const lrT = adamCfg.learningRate * (1 - step / numSteps);
  for (let i = 0; i < params.length; i++) {

    const p = params[i]!;

    adamState.m[i] = adamCfg.beta1 * adamState.m[i]! + (1 - adamCfg.beta1) * p.grad;
    adamState.v[i] = adamCfg.beta2 * adamState.v[i]! + (1 - adamCfg.beta2) * p.grad ** 2;

    const mHat = adamState.m[i]! / (1 - adamCfg.beta1 ** (step + 1));
    const vHat = adamState.v[i]! / (1 - adamCfg.beta2 ** (step + 1));

    p.data -= lrT * mHat / (vHat ** 0.5 + adamCfg.eps);
    p.grad = 0;

  }

  return loss.data;

}
