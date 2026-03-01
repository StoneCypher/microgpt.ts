/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm. Everything else is just efficiency.
 *
 * Converted from Karpathy's microgpt.py.
 *
 * @module microgpt
 */

// ---------------------------------------------------------------------------
// Seeded PRNG
// ---------------------------------------------------------------------------

// A deterministic pseudo-random number generator using the mulberry32 algorithm,
// with Box-Muller transform for Gaussian sampling and weighted random choices.
export class SeededRandom {

  private state: number;

  // Create a new SeededRandom with the given integer seed. 
  constructor(seed: number) {
    this.state = seed | 0;
  }

  // Return a uniform random number in [0, 1). 
  random(): number {
    this.state = (this.state + 0x6D2B79F5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Return a sample from a Gaussian distribution with the given mean and standard deviation. 
  gauss(mean = 0, std = 1): number {
    // Box-Muller transform
    const u1 = this.random();
    const u2 = this.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + z * std;
  }

  // Shuffle an array in-place (Fisher-Yates). 
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
  }

  // Return a single random index chosen from `weights` (treated as unnormalized probabilities).
  // Equivalent to Python's `random.choices(range(n), weights=weights, k=1)[0]`.
  choices(weights: number[]): number {
    let total = 0;
    for (const w of weights) total += w;
    let r = this.random() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i]!;
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}

// A scalar value that tracks its computation graph for automatic differentiation.
export class Value {

  data: number;     // Scalar value computed during the forward pass. 
  grad: number;     // Derivative of the loss with respect to this node, computed during backward. 

  private readonly _children: Value[];
  private readonly _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }

  /** Addition: `this + other`. */
  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  /** Multiplication: `this * other`. */
  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  /** Exponentiation: `this ** exponent` (exponent is a plain number). */
  pow(exponent: number): Value {
    return new Value(
      this.data ** exponent,
      [this],
      [exponent * this.data ** (exponent - 1)],
    );
  }

  // Natural logarithm. 
  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  // Exponential (e^x). 
  exp(): Value {
    const ex = Math.exp(this.data);
    return new Value(ex, [this], [ex]);
  }

  // Rectified linear unit: max(0, x). 
  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0],
    );
  }

  // Negation: -this. 
  neg(): Value {
    return this.mul(-1);
  }

  // Subtraction: `this - other`. 
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  // Division: `this / other`. 
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  // Backpropagate gradients through the computation graph rooted at this node. Sets `this.grad = 1` and accumulates gradients on all ancestors.
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i]!;
      for (let j = 0; j < v._children.length; j++) {
        v._children[j]!.grad += v._localGrads[j]! * v.grad;
      }
    }
  }
}

// A 2-D array of {@link Value} objects representing a weight matrix. 
export type Matrix = Value[][];

// String-keyed collection of weight matrices. 
export type StateDict = Record<string, Matrix>;

// Linear (fully-connected) layer: multiply weight matrix `w` by input vector `x`.
export function linear(x: Value[], w: Matrix): Value[] {
  return w.map(row => {
    let acc = row[0]!.mul(x[0]!);
    for (let i = 1; i < row.length; i++) {
      acc = acc.add(row[i]!.mul(x[i]!));
    }
    return acc;
  });
}

// Numerically stable softmax over a vector of {@link Value} objects.
export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map(v => v.data));
  const exps = logits.map(v => v.sub(maxVal).exp());
  let total = exps[0]!;
  for (let i = 1; i < exps.length; i++) {
    total = total.add(exps[i]!);
  }
  return exps.map(e => e.div(total));
}

// Root-mean-square normalization.
export function rmsnorm(x: Value[]): Value[] {
  let ms = x[0]!.mul(x[0]!);
  for (let i = 1; i < x.length; i++) {
    ms = ms.add(x[i]!.mul(x[i]!));
  }
  ms = ms.div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map(xi => xi.mul(scale));
}

// Hyperparameters for the GPT model.
export interface GPTConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
}

// Create and randomly initialize a state dict (all model parameters) for a GPT with the given configuration and vocabulary size.
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

// Run a single forward pass of the GPT model for one token position.
// This follows GPT-2 with minor differences: RMSNorm instead of LayerNorm, no biases, and ReLU instead of GeLU.
export function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  stateDict: StateDict,
  config: GPTConfig,
): Value[] {

  const { nLayer, nHead } = config;
  const headDim = config.nEmbd / nHead;

  const tokEmb = stateDict['wte']![tokenId]!;
  const posEmb = stateDict['wpe']![posId]!;
  let x = tokEmb.map((t, i) => t.add(posEmb[i]!));
  x = rmsnorm(x);

  for (let li = 0; li < nLayer; li++) {
    // 1) Multi-head Attention block
    const xResidual = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]!);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]!);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]!);
    keys[li]!.push(k);
    values[li]!.push(v);

    const xAttn: Value[] = [];
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      const qH = q.slice(hs, hs + headDim);
      const kH = keys[li]!.map(ki => ki.slice(hs, hs + headDim));
      const vH = values[li]!.map(vi => vi.slice(hs, hs + headDim));

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

    // 2) MLP block
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]!);
    x = x.map(xi => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]!);
    x = x.map((a, i) => a.add(xResidual2[i]!));
  }

  return linear(x, stateDict['lm_head']!);
}

// Build a character-level tokenizer from a list of documents.
export function buildTokenizer(docs: string[]): {
  uchars: string[];
  BOS: number;
  vocabSize: number;
} {
  const charSet = new Set<string>();
  for (const doc of docs) {
    for (const ch of doc) {
      charSet.add(ch);
    }
  }
  const uchars = [...charSet].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  return { uchars, BOS, vocabSize };
}

// Tokenize a document string into an array of token ids, surrounded by BOS.
export function tokenize(doc: string, uchars: string[], BOS: number): number[] {
  return [BOS, ...Array.from(doc).map(ch => uchars.indexOf(ch)), BOS];
}

// Configuration for the Adam optimizer.
export interface AdamConfig {
  learningRate: number;
  beta1: number;
  beta2: number;
  eps: number;
}

// Mutable Adam optimizer state (first and second moment buffers). 
export interface AdamState {
  m: number[];
  v: number[];
}

// Create a zeroed Adam state for `n` parameters. 
export const createAdamState = (n: number): AdamState => 
  ({ m: new Array<number>(n).fill(0), v: new Array<number>(n).fill(0) });

// Perform a single training step: forward pass, loss computation, backward pass, and Adam parameter update.
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

  const n = Math.min(config.blockSize, tokens.length - 1);
  const keys: Value[][][] = Array.from({ length: config.nLayer }, () => []);
  const valuesCache: Value[][][] = Array.from({ length: config.nLayer }, () => []);

  const losses: Value[] = [];
  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId]!;
    const targetId = tokens[posId + 1]!;
    const logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config);
    const probs = softmax(logits);
    losses.push(probs[targetId]!.log().neg());
  }

  let loss = losses[0]!;
  for (let i = 1; i < losses.length; i++) { loss = loss.add(losses[i]!); }
  loss = loss.div(n);
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

// Generate a single sequence from the model by autoregressively sampling tokens.
export function generate(
  stateDict: StateDict,
  config: GPTConfig,
  uchars: string[],
  BOS: number,
  rng: SeededRandom,
  temperature = 0.5,
): string {

  const keys: Value[][][] = Array.from({ length: config.nLayer }, () => []);
  const valuesCache: Value[][][] = Array.from({ length: config.nLayer }, () => []);
  let tokenId = BOS;
  const sample: string[] = [];

  for (let posId = 0; posId < config.blockSize; posId++) {
    const logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config);
    const scaledLogits = logits.map(l => l.div(temperature));
    const probs = softmax(scaledLogits);
    tokenId = rng.choices(probs.map(p => p.data));
    if (tokenId === BOS) break;
    sample.push(uchars[tokenId]!);
  }

  return sample.join('');
}
