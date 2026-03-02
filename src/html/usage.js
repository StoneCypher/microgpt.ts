
import {                                                                                              SeededRandom,
  buildTokenizer,                                                                                     tokenize,                                                                                           createStateDict,                                                                                
  createAdamState,
  trainStep,
  generate,
} from './index.mjs';





export function train({ 
  seed = 42, docs,
  nLayer = 1, nEmbd = 16, blockSize = 16, nHead = 4,
  learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, eps = 1e-8, temperature = 0.5,
  numSteps = 1000, commentNthStep = 100, sampleCount = 10,
}) {

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
    const doc = docs[step % docs.length],
          tokens = tokenize(doc, uchars, BOS),
          loss = trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, numSteps);  
    if ((step % commentNthStep) === 0) console.log(`step ${step} | loss ${loss.toFixed(4)}`);
  }

  // 5. Generate
  for (let i = 0; i < sampleCount; i++) {
    const sample = generate(stateDict, config, uchars, BOS, rng, temperature);
    console.log(sample);
  }
  
}
