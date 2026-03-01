export class Value {
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
