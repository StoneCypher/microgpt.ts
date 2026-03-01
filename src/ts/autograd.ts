
/**
 * Scalar autograd engine for automatic differentiation.
 *
 * @module autograd
 */





/**
 * A scalar value that tracks its computation graph for automatic differentiation.
 *
 * Because TypeScript has no operator overloading, arithmetic is performed via
 * methods: `a.add(b)`, `a.mul(b)`, `a.pow(n)`, etc.
 */

export class Value {

  /** Scalar value computed during the forward pass. */
  data: number;
  /** Derivative of the loss with respect to this node, computed during backward. */
  grad: number;

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

  /** Natural logarithm. */
  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  /** Exponential (e^x). */
  exp(): Value {
    const ex = Math.exp(this.data);
    return new Value(ex, [this], [ex]);
  }

  /** Rectified linear unit: max(0, x). */
  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0],
    );
  }

  /** Negation: -this. */
  neg(): Value {
    return this.mul(-1);
  }

  /** Subtraction: `this - other`. */
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  /** Division: `this / other`. */
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  /**
   * Backpropagate gradients through the computation graph rooted at this node.
   * Sets `this.grad = 1` and accumulates gradients on all ancestors.
   */
  backward(): void {

    const topo: Value[] = [],
          visited = new Set<Value>();

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
