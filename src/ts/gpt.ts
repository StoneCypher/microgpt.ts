/**
 * GPT model initialization and forward pass.
 *
 * @module gpt
 */

import { Value } from './autograd';
import { SeededRandom } from './rng';
import { linear, softmax, rms_norm } from './nn_helpers';
import type { Matrix, StateDict, GPTConfig } from './types';





/**
 * Create and randomly initialize a state dict (all model parameters) for a GPT
 * with the given configuration and vocabulary size.
 *
 * @param vocabSize  Total number of unique tokens (including the BOS token).
 * @param config     Model hyperparameters.
 * @param rng        Seeded random number generator.
 * @param std        Standard deviation for weight initialization (default 0.08).
 * 
 * @returns  An object containing `stateDict` and a flat `params` array.
 */

export function createStateDict(
  vocabSize: number,
  config: GPTConfig,
  rng: SeededRandom,
  std = 0.08,
): { stateDict: StateDict; params: Value[] } {

  const matrix = (nout: number, nin: number): Matrix =>
    Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => new Value(rng.gauss(0, std))),
    );

  const stateDict: StateDict = {
    wte: matrix(vocabSize, config.nEmbd),
    wpe: matrix(config.blockSize, config.nEmbd),
    lm_head: matrix(vocabSize, config.nEmbd),
  };

  for (let i = 0; i < config.nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wv`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wo`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(config.nEmbd, 4 * config.nEmbd);
  }

  const params: Value[] = [];
  for (const mat of Object.values(stateDict)) {
    for (const row of mat) {
      for (const p of row) {
        params.push(p);
      }
    }
  }

  return { stateDict, params };
}



/**
 * Run a single forward pass of the GPT model for one token position.
 *
 * This follows GPT-2 with minor differences: RMSNorm instead of LayerNorm,
 * no biases, and ReLU instead of GeLU.
 *
 * @param tokenId    Token id to embed.
 * @param posId      Position index within the sequence.
 * @param keys       Per-layer KV cache for keys (mutated: new key appended).
 * @param values     Per-layer KV cache for values (mutated: new value appended).
 * @param stateDict  Model weights.
 * @param config     Model hyperparameters.
 * 
 * @returns  Logits vector of length `vocabSize`.
 */

export function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  stateDict: StateDict,
  config: GPTConfig,
): Value[] {

  const { nLayer, nHead } = config,
        headDim = config.nEmbd / nHead;

  const tokEmb = stateDict['wte']![tokenId]!,
        posEmb = stateDict['wpe']![posId]!;

  let x = tokEmb.map((t, i) => t.add(posEmb[i]!));
  x = rms_norm(x);

  for (let li = 0; li < nLayer; li++) {
    // 1. Multi-head Attention block
    const xResidual = x;
    x = rms_norm(x);

    const q = linear(x, stateDict[`layer${li}.attn_wq`]!),
          k = linear(x, stateDict[`layer${li}.attn_wk`]!),
          v = linear(x, stateDict[`layer${li}.attn_wv`]!);

    keys[li]!.push(k);
    values[li]!.push(v);

    const xAttn: Value[] = [];
    for (let h = 0; h < nHead; h++) {

      const hs = h * headDim,
            qH = q.slice(hs, hs + headDim),
            kH = keys[li]!.map(ki => ki.slice(hs, hs + headDim)),
            vH = values[li]!.map(vi => vi.slice(hs, hs + headDim));

      const attnLogits: Value[] = [];
      for (let t = 0; t < kH.length; t++) {
        let dot = qH[0]!.mul(kH[t]![0]!);
        for (let j = 1; j < headDim; j++) {
          dot = dot.add(qH[j]!.mul(kH[t]![j]!));
        }
        attnLogits.push(dot.div(headDim ** 0.5));
      }

      const attnWeights = softmax(attnLogits);

      for (let j = 0; j < headDim; j++) {
        let sum = attnWeights[0]!.mul(vH[0]![j]!);
        for (let t = 1; t < vH.length; t++) {
          sum = sum.add(attnWeights[t]!.mul(vH[t]![j]!));
        }
        xAttn.push(sum);
      }
    }

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]!);
    x = x.map((a, i) => a.add(xResidual[i]!));

    // 2. MLP block
    const xResidual2 = x;
    x = rms_norm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]!);
    x = x.map(xi => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]!);
    x = x.map((a, i) => a.add(xResidual2[i]!));
  }

  return linear(x, stateDict['lm_head']!);

}
