import { describe, expect, it } from "vitest";
import { gamma, chisq } from "../src/stats/distribution.js";

// Regression: the lower-incomplete-gamma series was missing its 1/s leading
// factor, so gamma.cdf (and chisq.cdf) came out scaled by `shape`. Correct for
// shape=1 (exponential), wrong for shape>1.
describe("gamma / chi-squared CDF", () => {
  it("matches the exponential CDF (shape=1)", () => {
    expect(gamma.cdf(1, { shape: 1, scale: 1 })).toBeCloseTo(1 - Math.exp(-1), 6);
    expect(gamma.cdf(2, { shape: 1, scale: 1 })).toBeCloseTo(1 - Math.exp(-2), 6);
  });

  it("returns valid probabilities in [0,1] for shape > 1", () => {
    for (const s of [1.5, 2, 5.5, 10]) {
      const v = gamma.cdf(s * 2, { shape: s, scale: 1 });
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("matches known chi-squared 0.95 / 0.99 quantiles", () => {
    // Critical values: χ²(df) at p=0.95
    expect(chisq.cdf(5.991, { df: 2 })).toBeCloseTo(0.95, 3);
    expect(chisq.cdf(9.488, { df: 4 })).toBeCloseTo(0.95, 3);
    expect(chisq.cdf(18.307, { df: 10 })).toBeCloseTo(0.95, 3);
    expect(chisq.cdf(19.675, { df: 11 })).toBeCloseTo(0.95, 3);
    // p=0.99
    expect(chisq.cdf(24.725, { df: 11 })).toBeCloseTo(0.99, 3);
  });

  it("is monotone increasing", () => {
    let prev = -1;
    for (let x = 0.5; x <= 30; x += 0.5) {
      const v = chisq.cdf(x, { df: 11 });
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });
});
