export function linear(x, w) {
    return w.map(row => {
        let acc = row[0].mul(x[0]);
        for (let i = 1; i < row.length; i++) {
            acc = acc.add(row[i].mul(x[i]));
        }
        return acc;
    });
}
export function softmax(logits) {
    const maxVal = Math.max(...logits.map(v => v.data)), exps = logits.map(v => v.sub(maxVal).exp());
    let total = exps[0];
    for (let i = 1; i < exps.length; i++) {
        total = total.add(exps[i]);
    }
    return exps.map(e => e.div(total));
}
export function rms_norm(x) {
    let ms = x[0].mul(x[0]);
    for (let i = 1; i < x.length; i++) {
        ms = ms.add(x[i].mul(x[i]));
    }
    ms = ms.div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map(xi => xi.mul(scale));
}
