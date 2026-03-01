
describe('test rig is running', () => {
  test('arithmetic works', () => {
    expect(2+2).toBe(4);
  });
});

import {
  Value,
  SeededRandom,
  linear,
  softmax,
  rms_norm,
  buildTokenizer,
  tokenize,
  createStateDict,
  gpt,
  trainStep,
  createAdamState,
  generate,
} from '../index';

// ---------------------------------------------------------------------------
// SeededRandom
// ---------------------------------------------------------------------------

describe('SeededRandom', () => {

  test('produces deterministic output for the same seed', () => {
    const a = new SeededRandom(42);
    const b = new SeededRandom(42);
    for (let i = 0; i < 100; i++) {
      expect(a.random()).toBe(b.random());
    }
  });

  test('random() returns values in [0, 1)', () => {
    const rng = new SeededRandom(123);
    for (let i = 0; i < 1000; i++) {
      const v = rng.random();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  test('gauss() produces finite numbers', () => {
    const rng = new SeededRandom(7);
    for (let i = 0; i < 100; i++) {
      expect(Number.isFinite(rng.gauss(0, 1))).toBe(true);
    }
  });

  test('shuffle() is a permutation', () => {
    const rng = new SeededRandom(99);
    const arr = [1, 2, 3, 4, 5, 6, 7, 8];
    const sorted = [...arr];
    rng.shuffle(arr);
    expect(arr.sort((a, b) => a - b)).toEqual(sorted);
  });

  test('choices() respects weights', () => {
    const rng = new SeededRandom(0);
    const counts = [0, 0, 0];
    const weights = [1, 0, 0]; // always pick index 0
    for (let i = 0; i < 100; i++) {
      counts[rng.choices(weights)]!++;
    }
    expect(counts[0]).toBe(100);
  });

  test('choices() fallback returns last index on floating-point overshoot', () => {
    const rng = new SeededRandom(0);
    // Mock random() to return just above 1.0 so r stays positive after
    // subtracting all weights — exercises the safety fallback on line 58
    vi.spyOn(rng, 'random').mockReturnValue(1 + Number.EPSILON);
    const result = rng.choices([0.5, 0.5]);
    expect(result).toBe(1);
    vi.restoreAllMocks();
  });
});

// ---------------------------------------------------------------------------
// Value — arithmetic
// ---------------------------------------------------------------------------

describe('Value arithmetic', () => {

  test('add two Values', () => {
    const a = new Value(3);
    const b = new Value(4);
    expect(a.add(b).data).toBeCloseTo(7);
  });

  test('add Value and number', () => {
    expect(new Value(3).add(2).data).toBeCloseTo(5);
  });

  test('mul two Values', () => {
    expect(new Value(3).mul(new Value(4)).data).toBeCloseTo(12);
  });

  test('mul Value and number', () => {
    expect(new Value(3).mul(5).data).toBeCloseTo(15);
  });

  test('pow', () => {
    expect(new Value(2).pow(3).data).toBeCloseTo(8);
  });

  test('neg', () => {
    expect(new Value(5).neg().data).toBeCloseTo(-5);
  });

  test('sub', () => {
    expect(new Value(10).sub(new Value(3)).data).toBeCloseTo(7);
  });

  test('div', () => {
    expect(new Value(10).div(new Value(4)).data).toBeCloseTo(2.5);
  });

  test('log', () => {
    expect(new Value(Math.E).log().data).toBeCloseTo(1);
  });

  test('exp', () => {
    expect(new Value(1).exp().data).toBeCloseTo(Math.E);
  });

  test('relu positive', () => {
    expect(new Value(3).relu().data).toBeCloseTo(3);
  });

  test('relu negative', () => {
    expect(new Value(-3).relu().data).toBeCloseTo(0);
  });
});

// ---------------------------------------------------------------------------
// Value — backward
// ---------------------------------------------------------------------------

describe('Value backward', () => {

  test('gradient of a*b + c', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = new Value(5);
    const out = a.mul(b).add(c); // 2*3 + 5 = 11
    out.backward();
    expect(out.data).toBeCloseTo(11);
    expect(a.grad).toBeCloseTo(3); // d/da = b
    expect(b.grad).toBeCloseTo(2); // d/db = a
    expect(c.grad).toBeCloseTo(1); // d/dc = 1
  });

  test('gradient of a**2', () => {
    const a = new Value(4);
    const out = a.pow(2); // 16
    out.backward();
    expect(a.grad).toBeCloseTo(8); // d/da = 2*a = 8
  });

  test('gradient of log(a)', () => {
    const a = new Value(2);
    const out = a.log();
    out.backward();
    expect(a.grad).toBeCloseTo(0.5); // d/da = 1/a
  });

  test('gradient of exp(a)', () => {
    const a = new Value(1);
    const out = a.exp();
    out.backward();
    expect(a.grad).toBeCloseTo(Math.E); // d/da = exp(a)
  });

  test('gradient of relu', () => {
    const a = new Value(3);
    const outPos = a.relu();
    outPos.backward();
    expect(a.grad).toBeCloseTo(1);

    const b = new Value(-3);
    const outNeg = b.relu();
    outNeg.backward();
    expect(b.grad).toBeCloseTo(0);
  });

  test('gradient through division', () => {
    const a = new Value(6);
    const b = new Value(3);
    const out = a.div(b); // 6/3 = 2
    out.backward();
    expect(out.data).toBeCloseTo(2);
    expect(a.grad).toBeCloseTo(1 / 3);    // d/da = 1/b
    expect(b.grad).toBeCloseTo(-6 / 9);   // d/db = -a/b^2
  });
});

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

describe('linear', () => {
  test('matrix-vector multiply', () => {
    const x = [new Value(1), new Value(2)];
    const w = [
      [new Value(3), new Value(4)],
      [new Value(5), new Value(6)],
    ];
    const y = linear(x, w);
    expect(y).toHaveLength(2);
    expect(y[0]!.data).toBeCloseTo(11); // 3*1 + 4*2
    expect(y[1]!.data).toBeCloseTo(17); // 5*1 + 6*2
  });
});

describe('softmax', () => {
  test('outputs sum to 1', () => {
    const logits = [new Value(1), new Value(2), new Value(3)];
    const probs = softmax(logits);
    const total = probs.reduce((s, p) => s + p.data, 0);
    expect(total).toBeCloseTo(1);
  });

  test('all values are in [0, 1]', () => {
    const logits = [new Value(-5), new Value(0), new Value(5)];
    const probs = softmax(logits);
    for (const p of probs) {
      expect(p.data).toBeGreaterThanOrEqual(0);
      expect(p.data).toBeLessThanOrEqual(1);
    }
  });

  test('highest logit gets highest probability', () => {
    const probs = softmax([new Value(1), new Value(10), new Value(2)]);
    expect(probs[1]!.data).toBeGreaterThan(probs[0]!.data);
    expect(probs[1]!.data).toBeGreaterThan(probs[2]!.data);
  });
});

describe('rms_norm', () => {
  test('output has approximately unit RMS', () => {
    const x = [new Value(3), new Value(4), new Value(5)];
    const normed = rms_norm(x);
    const rms = Math.sqrt(normed.reduce((s, v) => s + v.data ** 2, 0) / normed.length);
    expect(rms).toBeCloseTo(1, 1);
  });
});

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

describe('tokenizer', () => {
  test('buildTokenizer returns sorted unique chars', () => {
    const { uchars, BOS, vocabSize } = buildTokenizer(['bac', 'cab']);
    expect(uchars).toEqual(['a', 'b', 'c']);
    expect(BOS).toBe(3);
    expect(vocabSize).toBe(4);
  });

  test('tokenize wraps doc in BOS', () => {
    const { uchars, BOS } = buildTokenizer(['ab']);
    const tokens = tokenize('ab', uchars, BOS);
    expect(tokens).toEqual([BOS, 0, 1, BOS]);
  });
});

// ---------------------------------------------------------------------------
// GPT integration
// ---------------------------------------------------------------------------

describe('GPT integration', () => {

  // Use a tiny model so tests run fast
  const config = { nLayer: 1, nEmbd: 4, blockSize: 4, nHead: 2 };
  const docs = ['abc', 'bca', 'cab'];
  const { uchars, BOS, vocabSize } = buildTokenizer(docs);
  const adamCfg = { learningRate: 0.01, beta1: 0.85, beta2: 0.99, eps: 1e-8 };

  test('gpt forward returns logits of correct length', () => {
    const rng = new SeededRandom(42);
    const { stateDict } = createStateDict(vocabSize, config, rng);
    const keys = Array.from({ length: config.nLayer }, (): Value[][] => []);
    const vals = Array.from({ length: config.nLayer }, (): Value[][] => []);
    const logits = gpt(BOS, 0, keys, vals, stateDict, config);
    expect(logits).toHaveLength(vocabSize);
    for (const l of logits) {
      expect(Number.isFinite(l.data)).toBe(true);
    }
  });

  test('trainStep returns a finite loss', () => {
    const rng = new SeededRandom(42);
    const { stateDict, params } = createStateDict(vocabSize, config, rng);
    const adamState = createAdamState(params.length);
    const tokens = tokenize('abc', uchars, BOS);
    const loss = trainStep(tokens, params, stateDict, config, adamCfg, adamState, 0, 10);
    expect(Number.isFinite(loss)).toBe(true);
  });

  test('loss decreases after a few steps', () => {
    const rng = new SeededRandom(42);
    const { stateDict, params } = createStateDict(vocabSize, config, rng);
    const adamState = createAdamState(params.length);
    const tokens = tokenize('abc', uchars, BOS);

    const loss0 = trainStep(tokens, params, stateDict, config, adamCfg, adamState, 0, 20);
    let lastLoss = loss0;
    for (let step = 1; step < 20; step++) {
      lastLoss = trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, 20);
    }
    expect(lastLoss).toBeLessThan(loss0);
  });

  test('generate produces a string', () => {
    const rng = new SeededRandom(42);
    const { stateDict } = createStateDict(vocabSize, config, rng);
    const result = generate(stateDict, config, uchars, BOS, new SeededRandom(7));
    expect(typeof result).toBe('string');
  });
});
