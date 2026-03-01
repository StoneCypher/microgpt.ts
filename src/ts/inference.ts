/**
 * Inference / text generation for MicroGPT.
 *
 * @module inference
 */

import { Value } from './autograd';
import { SeededRandom } from './rng';
import { softmax } from './nn_helpers';
import { gpt } from './gpt';
import type { StateDict, GPTConfig } from './types';





/**
 * Generate a single sequence from the model by autoregressively sampling tokens.
 *
 * @param stateDict   Model weights.
 * @param config      Model hyperparameters.
 * @param uchars      Character vocabulary (index → character).
 * @param BOS         BOS token id.
 * @param rng         Seeded random number generator.
 * @param temperature Sampling temperature in (0, 1]; lower is less random.
 * 
 * @returns  The generated string.
 */

export function generate(
  stateDict: StateDict,
  config: GPTConfig,
  uchars: string[],
  BOS: number,
  rng: SeededRandom,
  temperature = 0.5,
): string {

  const keys: Value[][][] = Array.from({ length: config.nLayer }, () => []),
        valuesCache: Value[][][] = Array.from({ length: config.nLayer }, () => []),
        sample: string[] = [];

  let tokenId = BOS;

  for (let posId = 0; posId < config.blockSize; posId++) {

    const logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config),
          scaledLogits = logits.map(l => l.div(temperature)),
          probs = softmax(scaledLogits);

    tokenId = rng.choices(probs.map(p => p.data));
    if (tokenId === BOS) break;

    sample.push(uchars[tokenId]!);

  }

  return sample.join('');

}
