import { Value } from './autograd';
import type { Matrix } from './types';
export declare function linear(x: Value[], w: Matrix): Value[];
export declare function softmax(logits: Value[]): Value[];
export declare function rms_norm(x: Value[]): Value[];
