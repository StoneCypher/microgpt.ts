import { Value } from './autograd';
import type { StateDict, GPTConfig, AdamConfig, AdamState } from './types';
export declare function createAdamState(n: number): AdamState;
export declare function trainStep(tokens: number[], params: Value[], stateDict: StateDict, config: GPTConfig, adamCfg: AdamConfig, adamState: AdamState, step: number, numSteps: number): number;
