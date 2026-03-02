class Value {
    data;
    grad;
    _children;
    _localGrads;
    constructor(data, children = [], localGrads = []) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }
    add(other) {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data + o.data, [this, o], [1, 1]);
    }
    mul(other) {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data * o.data, [this, o], [o.data, this.data]);
    }
    pow(exponent) {
        return new Value(this.data ** exponent, [this], [exponent * this.data ** (exponent - 1)]);
    }
    log() {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }
    exp() {
        const ex = Math.exp(this.data);
        return new Value(ex, [this], [ex]);
    }
    relu() {
        return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
    }
    neg() {
        return this.mul(-1);
    }
    sub(other) {
        const o = other instanceof Value ? other : new Value(other);
        return this.add(o.neg());
    }
    div(other) {
        const o = other instanceof Value ? other : new Value(other);
        return this.mul(o.pow(-1));
    }
    backward() {
        const topo = [], visited = new Set();
        const buildTopo = (v) => {
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
            const v = topo[i];
            for (let j = 0; j < v._children.length; j++) {
                v._children[j].grad += v._localGrads[j] * v.grad;
            }
        }
    }
}

function linear(x, w) {
    return w.map(row => {
        let acc = row[0].mul(x[0]);
        for (let i = 1; i < row.length; i++) {
            acc = acc.add(row[i].mul(x[i]));
        }
        return acc;
    });
}
function softmax(logits) {
    const maxVal = Math.max(...logits.map(v => v.data)), exps = logits.map(v => v.sub(maxVal).exp());
    let total = exps[0];
    for (let i = 1; i < exps.length; i++) {
        total = total.add(exps[i]);
    }
    return exps.map(e => e.div(total));
}
function rms_norm(x) {
    let ms = x[0].mul(x[0]);
    for (let i = 1; i < x.length; i++) {
        ms = ms.add(x[i].mul(x[i]));
    }
    ms = ms.div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map(xi => xi.mul(scale));
}

function createStateDict(vocabSize, config, rng, std = 0.08) {
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
function gpt(tokenId, posId, keys, values, stateDict, config) {
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

function generate(stateDict, config, uchars, BOS, rng, temperature = 0.5) {
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

class SeededRandom {
    state;
    constructor(seed) {
        this.state = seed | 0;
    }
    random() {
        this.state = (this.state + 0x6D2B79F5) | 0;
        let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    gauss(mean = 0, std = 1) {
        const u1 = this.random() || Number.MIN_VALUE, u2 = this.random(), z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mean + z * std;
    }
    shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }
    choices(weights) {
        let total = 0;
        for (const w of weights)
            total += w;
        let r = this.random() * total;
        for (let i = 0; i < weights.length; i++) {
            r -= weights[i];
            if (r <= 0)
                return i;
        }
        return weights.length - 1;
    }
}

function buildTokenizer(docs) {
    const charSet = new Set();
    for (const doc of docs) {
        for (const ch of doc) {
            charSet.add(ch);
        }
    }
    const uchars = [...charSet].sort(), BOS = uchars.length, vocabSize = uchars.length + 1;
    return { uchars, BOS, vocabSize };
}
function tokenize(doc, uchars, BOS) {
    return [BOS, ...Array.from(doc ?? '').map(ch => uchars.indexOf(ch)), BOS];
}

function createAdamState(n) {
    return { m: new Array(n).fill(0), v: new Array(n).fill(0) };
}
function trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, numSteps) {
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

async function train({ seed = 42, docs, nLayer = 1, nEmbd = 16, blockSize = 16, nHead = 4, learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, eps = 1e-8, temperature = 0.5, numSteps = 1000, commentNthStep = 100, sampleCount = 10, onStep, signal, }) {
    if (docs === undefined) {
        throw new Error('Must define docs');
    }
    if (!(Array.isArray(docs))) {
        throw new Error('Docs must be an array of strings');
    }
    const rng = new SeededRandom(seed);
    rng.shuffle(docs);
    const { uchars, BOS, vocabSize } = buildTokenizer(docs);
    const config = { nLayer, nEmbd, blockSize, nHead }, { stateDict, params } = createStateDict(vocabSize, config, rng);
    const adamCfg = { learningRate, beta1, beta2, eps }, adamState = createAdamState(params.length);
    for (let step = 0; step < numSteps; step++) {
        if (signal?.aborted)
            break;
        const doc = docs[step % docs.length], tokens = tokenize(doc, uchars, BOS), loss = trainStep(tokens, params, stateDict, config, adamCfg, adamState, step, numSteps);
        const isComment = (step % commentNthStep) === 0;
        if (isComment)
            console.log(`step ${step} | loss ${loss.toFixed(4)}`);
        if (onStep) {
            const samples = [];
            if (isComment)
                for (let i = 0; i < sampleCount; i++)
                    samples.push(generate(stateDict, config, uchars, BOS, rng, temperature));
            onStep({ step, numSteps, loss, samples });
        }
        if (isComment)
            await new Promise(r => setTimeout(r, 0));
    }
    const finalSamples = [];
    for (let i = 0; i < sampleCount; i++) {
        const sample = generate(stateDict, config, uchars, BOS, rng, temperature);
        finalSamples.push(sample);
        console.log(sample);
    }
    return finalSamples;
}

export { SeededRandom, Value, buildTokenizer, createAdamState, createStateDict, generate, gpt, linear, rms_norm, softmax, tokenize, train, trainStep };
