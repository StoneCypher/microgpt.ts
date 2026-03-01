import { Value } from './autograd';
import { SeededRandom } from './rng';
import type { StateDict, GPTConfig } from './types';
export declare function createStateDict(vocabSize: number, config: GPTConfig, rng: SeededRandom, std?: number): {
    stateDict: StateDict;
    params: Value[];
};
export declare function gpt(tokenId: number, posId: number, keys: Value[][][], values: Value[][][], stateDict: StateDict, config: GPTConfig): Value[];
