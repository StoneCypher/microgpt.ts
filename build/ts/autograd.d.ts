export declare class Value {
    data: number;
    grad: number;
    private readonly _children;
    private readonly _localGrads;
    constructor(data: number, children?: Value[], localGrads?: number[]);
    add(other: Value | number): Value;
    mul(other: Value | number): Value;
    pow(exponent: number): Value;
    log(): Value;
    exp(): Value;
    relu(): Value;
    neg(): Value;
    sub(other: Value | number): Value;
    div(other: Value | number): Value;
    backward(): void;
}
