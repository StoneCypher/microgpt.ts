export declare class SeededRandom {
    private state;
    constructor(seed: number);
    random(): number;
    gauss(mean?: number, std?: number): number;
    shuffle<T>(arr: T[]): T[];
    choices(weights: number[]): number;
}
