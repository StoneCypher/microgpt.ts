/**
 * Neural-network helper functions.
 *
 * @module nn_helpers
 */

import { Value } from './autograd';
import type { Matrix } from './types';





/**
 * Linear (fully-connected) layer: multiply weight matrix `w` by input vector `x`.
 *
 * @param x  Input vector of length `nin`.
 * @param w  Weight matrix of shape `[nout][nin]`.
 * @returns  Output vector of length `nout`.
 */

export function linear(x: Value[], w: Matrix): Value[] {

  return w.map(row => {

    let acc = row[0]!.mul(x[0]!);

    for (let i = 1; i < row.length; i++) {
      acc = acc.add(row[i]!.mul(x[i]!));
    }

    return acc;

  });

}



/**
 * Numerically stable softmax over a vector of {@link Value} objects.
 *
 * @param logits  Input logits.
 * @returns  Probability distribution (Values that sum to 1).
 */

export function softmax(logits: Value[]): Value[] {

  const maxVal = Math.max(...logits.map(v => v.data)),
        exps = logits.map(v => v.sub(maxVal).exp());

  let total = exps[0]!;

  for (let i = 1; i < exps.length; i++) {
    total = total.add(exps[i]!);
  }

  return exps.map(e => e.div(total));

}



/**
 * Root-mean-square normalization.
 *
 * @param x  Input vector.
 * @returns  Normalized vector with approximately unit RMS.
 */

export function rms_norm(x: Value[]): Value[] {

  let ms = x[0]!.mul(x[0]!);

  for (let i = 1; i < x.length; i++) {
    ms = ms.add(x[i]!.mul(x[i]!));
  }

  ms = ms.div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);

  return x.map(xi => xi.mul(scale));

}
