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
export type TrainType = {
    seed: number;
    docs: string[];
    nLayer: number;
    nEmbd: number;
    blockSize: number;
    nHead: number;
    learningRate: number;
    beta1: number;
    beta2: number;
    eps: number;
    temperature: number;
    numSteps: number;
    commentNthStep: number;
    sampleCount: number;
    onStep?: (update: {
        step: number;
        numSteps: number;
        loss: number;
        samples: string[];
    }) => void;
    signal?: {
        aborted: boolean;
    };
};
