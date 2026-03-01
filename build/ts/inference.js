import { softmax } from './nn_helpers';
import { gpt } from './gpt';
export function generate(stateDict, config, uchars, BOS, rng, temperature = 0.5) {
    const keys = Array.from({ length: config.nLayer }, () => []), valuesCache = Array.from({ length: config.nLayer }, () => []), sample = [];
    let tokenId = BOS;
    for (let posId = 0; posId < config.blockSize; posId++) {
        const logits = gpt(tokenId, posId, keys, valuesCache, stateDict, config), scaledLogits = logits.map(l => l.div(temperature)), probs = softmax(scaledLogits);
        tokenId = rng.choices(probs.map(p => p.data));
        if (tokenId === BOS)
            break;
        sample.push(uchars[tokenId]);
    }
    return sample.join('');
}
