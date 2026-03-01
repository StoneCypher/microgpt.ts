import { SeededRandom } from './rng';
import type { StateDict, GPTConfig } from './types';
export declare function generate(stateDict: StateDict, config: GPTConfig, uchars: string[], BOS: number, rng: SeededRandom, temperature?: number): string;
