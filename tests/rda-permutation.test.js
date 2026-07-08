/**
 * Unit tests for the RDA global permutation test (rda.permutationTest /
 * RDA#anova). These run without R: they check internal consistency, rank-aware
 * degrees of freedom, reproducibility, and that the test separates a genuine
 * signal from noise. The vegan cross-check lives in r-comparison.test.js.
 */

import { describe, it, expect } from 'vitest';
import * as rda from '../src/mva/rda.js';
import { RDA } from '../src/mva/estimators/RDA.js';

// Small seeded generator so the fixtures are fully deterministic.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Box-Muller standard normal from a uniform generator.
function randn(u) {
  return Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
}

/** n rows, p predictors, q responses; Y = X B + noise (B present iff signal). */
function makeData(seed, n, p, q, signal) {
  const u = rng(seed);
  const X = Array.from({ length: n }, () => Array.from({ length: p }, () => randn(u)));
  const B = Array.from({ length: p }, () => Array.from({ length: q }, () => (signal ? randn(u) : 0)));
  const Y = X.map((row) => {
    const yhat = Array.from({ length: q }, (_, j) =>
      row.reduce((s, xk, k) => s + xk * B[k][j], 0));
    return yhat.map((v) => v + 0.7 * randn(u));
  });
  return { X, Y };
}

describe('rda.permutationTest - internal consistency', () => {
  it('reports F equal to the value computed at fit time', () => {
    const { X, Y } = makeData(1, 60, 3, 4, true);
    const model = rda.fit(Y, X, { scale: true });
    const av = rda.permutationTest(model, { permutations: 199, seed: 7 });
    expect(av.pseudoF).toBeCloseTo(model.pseudoF, 9);
    expect(av.dfModel).toBe(model.dfModel);
    expect(av.dfResidual).toBe(model.dfResidual);
  });

  it('decomposes inertia consistently (constrained + residual = total)', () => {
    const { X, Y } = makeData(2, 60, 3, 4, true);
    const av = rda.permutationTest(rda.fit(Y, X, { scale: true }), { permutations: 99, seed: 1 });
    expect(av.constrainedInertia + av.residualInertia).toBeCloseTo(av.totalInertia, 9);
    expect(av.constrainedProportion).toBeCloseTo(av.constrainedInertia / av.totalInertia, 9);
    expect(av.pValue).toBeGreaterThanOrEqual(1 / (av.permutations + 1));
    expect(av.pValue).toBeLessThanOrEqual(1);
  });

  it('is reproducible for a fixed seed and varies with the seed', () => {
    const { X, Y } = makeData(3, 50, 3, 3, true);
    const model = rda.fit(Y, X, { scale: true });
    const a = rda.permutationTest(model, { permutations: 299, seed: 42 });
    const b = rda.permutationTest(model, { permutations: 299, seed: 42 });
    const c = rda.permutationTest(model, { permutations: 299, seed: 43 });
    expect(a.pValue).toBe(b.pValue);
    expect(a.pseudoF).toBe(b.pseudoF); // F is data-only, seed-independent
    expect(c.pseudoF).toBe(a.pseudoF);
  });
});

describe('rda.permutationTest - rank-aware degrees of freedom', () => {
  it('uses the numerical rank of the constraints, not the column count', () => {
    // Add a predictor that is an exact linear combination of two others: the
    // rank is 3, so dfModel must be 3 (not 4).
    const { X, Y } = makeData(4, 60, 3, 4, true);
    const Xcollinear = X.map((row) => [...row, 2 * row[0] - row[1]]);
    const model = rda.fit(Y, Xcollinear, { scale: true });
    expect(model.dfModel).toBe(3);
    expect(model.dfResidual).toBe(60 - 3 - 1);
  });
});

describe('rda.permutationTest - separates signal from noise', () => {
  it('flags a real constraint as significant', () => {
    const { X, Y } = makeData(10, 80, 3, 4, true);
    const av = rda.permutationTest(rda.fit(Y, X, { scale: true }), { permutations: 499, seed: 5 });
    expect(av.pseudoF).toBeGreaterThan(2);
    expect(av.pValue).toBeLessThan(0.05);
  });

  it('does not flag pure noise (F near 1, p not tiny)', () => {
    const { X, Y } = makeData(11, 80, 3, 4, false);
    const av = rda.permutationTest(rda.fit(Y, X, { scale: true }), { permutations: 499, seed: 6 });
    expect(av.pseudoF).toBeLessThan(2);
    expect(av.pValue).toBeGreaterThan(0.05);
  });
});

describe('RDA estimator - anova()/permutationTest()', () => {
  it('exposes the test through the class and matches the functional API', () => {
    const { X, Y } = makeData(12, 50, 3, 3, true);
    const est = new RDA({ scale: true }).fit(Y, X);
    const viaClass = est.anova({ permutations: 199, seed: 9 });
    const viaFn = rda.permutationTest(est.model, { permutations: 199, seed: 9 });
    expect(viaClass.pValue).toBe(viaFn.pValue);
    expect(viaClass.pseudoF).toBeCloseTo(viaFn.pseudoF, 9);
  });

  it('throws before fitting', () => {
    expect(() => new RDA({ scale: true }).permutationTest()).toThrow();
  });
});
