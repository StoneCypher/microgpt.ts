import { Value } from './autograd';
import { linear, softmax, rms_norm } from './nn_helpers';
export function createStateDict(vocabSize, config, rng, std = 0.08) {
    const matrix = (nout, nin) => Array.from({ length: nout }, () => Array.from({ length: nin }, () => new Value(rng.gauss(0, std))));
    const stateDict = {
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
    const params = [];
    for (const mat of Object.values(stateDict)) {
        for (const row of mat) {
            for (const p of row) {
                params.push(p);
            }
        }
    }
    return { stateDict, params };
}
export function gpt(tokenId, posId, keys, values, stateDict, config) {
    const { nLayer, nHead } = config, headDim = config.nEmbd / nHead;
    const tokEmb = stateDict['wte'][tokenId], posEmb = stateDict['wpe'][posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rms_norm(x);
    for (let li = 0; li < nLayer; li++) {
        const xResidual = x;
        x = rms_norm(x);
        const q = linear(x, stateDict[`layer${li}.attn_wq`]), k = linear(x, stateDict[`layer${li}.attn_wk`]), v = linear(x, stateDict[`layer${li}.attn_wv`]);
        keys[li].push(k);
        values[li].push(v);
        const xAttn = [];
        for (let h = 0; h < nHead; h++) {
            const hs = h * headDim, qH = q.slice(hs, hs + headDim), kH = keys[li].map(ki => ki.slice(hs, hs + headDim)), vH = values[li].map(vi => vi.slice(hs, hs + headDim));
            const attnLogits = [];
            for (let t = 0; t < kH.length; t++) {
                let dot = qH[0].mul(kH[t][0]);
                for (let j = 1; j < headDim; j++) {
                    dot = dot.add(qH[j].mul(kH[t][j]));
                }
                attnLogits.push(dot.div(headDim ** 0.5));
            }
            const attnWeights = softmax(attnLogits);
            for (let j = 0; j < headDim; j++) {
                let sum = attnWeights[0].mul(vH[0][j]);
                for (let t = 1; t < vH.length; t++) {
                    sum = sum.add(attnWeights[t].mul(vH[t][j]));
                }
                xAttn.push(sum);
            }
        }
        x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
        x = x.map((a, i) => a.add(xResidual[i]));
        const xResidual2 = x;
        x = rms_norm(x);
        x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
        x = x.map(xi => xi.relu());
        x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
        x = x.map((a, i) => a.add(xResidual2[i]));
    }
    return linear(x, stateDict['lm_head']);
}
