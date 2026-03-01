import { softmax } from './nn_helpers';
import { gpt } from './gpt';
export function createAdamState(n) {
    return { m: new Array(n).fill(0), v: new Array(n).fill(0) };
}
export function trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, numSteps) {
    const n = Math.min(config.blockSize, tokens.length - 1), keys = Array.from({ length: config.nLayer }, () => []), valuesCache = Array.from({ length: config.nLayer }, () => []);
    const losses = [];
    for (let posId = 0; posId < n; posId++) {
        const tokenId = tokens[posId], targetId = tokens[posId + 1], logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config), probs = softmax(logits);
        losses.push(probs[targetId].log().neg());
    }
    let loss = losses[0];
    for (let i = 1; i < losses.length; i++) {
        loss = loss.add(losses[i]);
    }
    loss = loss.div(n);
    loss.backward();
    const lrT = adamCfg.learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
        const p = params[i];
        adamState.m[i] = adamCfg.beta1 * adamState.m[i] + (1 - adamCfg.beta1) * p.grad;
        adamState.v[i] = adamCfg.beta2 * adamState.v[i] + (1 - adamCfg.beta2) * p.grad ** 2;
        const mHat = adamState.m[i] / (1 - adamCfg.beta1 ** (step + 1));
        const vHat = adamState.v[i] / (1 - adamCfg.beta2 ** (step + 1));
        p.data -= lrT * mHat / (vHat ** 0.5 + adamCfg.eps);
        p.grad = 0;
    }
    return loss.data;
}
