import type { Value } from './autograd';
export type Matrix = Value[][];
export type StateDict = Record<string, Matrix>;
export interface GPTConfig {
    nLayer: number;
    nEmbd: number;
    blockSize: number;
    nHead: number;
}
export interface AdamConfig {
    learningRate: number;
    beta1: number;
    beta2: number;
    eps: number;
}
export interface AdamState {
    m: number[];
    v: number[];
}
