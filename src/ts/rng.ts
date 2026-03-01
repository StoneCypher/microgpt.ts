/**
 * Deterministic seeded pseudo-random number generator.
 *
 * @module rng
 */





/**
 * A deterministic pseudo-random number generator using the mulberry32 algorithm,
 * with Box-Muller transform for Gaussian sampling and weighted random choices.
 */

export class SeededRandom {

  private state: number;


  
  /** Create a new SeededRandom with the given integer seed. */
  constructor(seed: number) {
    this.state = seed | 0;
  }



  /** Return a uniform random number in [0, 1). */
  random(): number {
  
    this.state = (this.state + 0x6D2B79F5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  
  }



  /** Return a sample from a Gaussian distribution with the given mean and standard deviation. */
  gauss(mean = 0, std = 1): number {
    // Box-Muller transform
  
    const u1 = this.random(),
          u2 = this.random(),
          z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  
    return mean + z * std;
  
  }



  /** Shuffle an array in-place (Fisher-Yates). */
  shuffle<T>(arr: T[]): T[] {

    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }

    return arr;

  }



  /**
   * Return a single random index chosen from `weights` (treated as unnormalized probabilities).
   * Equivalent to Python's `random.choices(range(n), weights=weights, k=1)[0]`.
   */

  choices(weights: number[]): number {

    let total = 0;
    for (const w of weights) total += w;

    let r = this.random() * total;

    for (let i = 0; i < weights.length; i++) {
      r -= weights[i]!;
      if (r <= 0) return i;
    }

    return weights.length - 1;

  }

}
