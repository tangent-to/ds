/**
 * Analytic marginal-likelihood gradient for the Matérn GP optimizer.
 * The gradient supplied to L-BFGS must match finite differences of the negative
 * log marginal likelihood (w.r.t. log-hyperparameters), and `optimize: true`
 * must find hyperparameters at least as good as the un-optimized start.
 */

import { describe, it, expect } from 'vitest';
import { GaussianProcessRegressor } from '../src/ml/estimators/GaussianProcessRegressor.js';
import { Matern, SumKernel, WhiteKernel } from '../src/ml/kernels/index.js';

function makeData(seed, n, D) {
  let s = seed >>> 0;
  const u = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const randn = () => Math.sqrt(-2 * Math.log(u() || 1e-9)) * Math.cos(2 * Math.PI * u());
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const x = Array.from({ length: D }, () => randn());
    X.push(x);
    y.push(2 + 1.5 * x[0] - 0.8 * x[1] + 0.3 * (x[2] || 0) * (x[2] || 0) + 0.2 * randn());
  }
  return { X, y };
}

// Rebuild the hyperparameter accessor list the optimizer uses, in kernel order:
// per Matérn [length scales…, variance], per WhiteKernel [noiseLevel]. `alpha`
// is NOT in the list — it is fixed, known noise, never a hyperparameter.
function hyperList(gp) {
  const hs = [];
  const visit = (k) => {
    if (k instanceof SumKernel) { k.kernels.forEach(visit); return; }
    if (k instanceof WhiteKernel) {
      hs.push({ get: () => k.noiseLevel, set: (v) => { k.noiseLevel = v; }, min: 1e-10 });
      return;
    }
    if (Array.isArray(k.lengthScale)) {
      k.lengthScale.forEach((_, i) => hs.push({ get: () => k.lengthScale[i], set: (v) => { k.lengthScale[i] = v; }, min: 1e-5 }));
    } else {
      hs.push({ get: () => k.lengthScale, set: (v) => { k.lengthScale = v; }, min: 1e-5 });
    }
    hs.push({ get: () => k.variance, set: (v) => { k.variance = v; }, min: 1e-8 });
  };
  visit(gp.kernel);
  return hs;
}

// The analytic path covers a bare Matérn and a Matérn summed with a WhiteKernel
// in either order; the gradient slots must follow the kernel's own ordering.
const COMPOSITIONS = [
  ['bare Matérn', (m) => m, (D) => D + 1],
  ['Matérn + WhiteKernel', (m) => new SumKernel({ kernels: [m, new WhiteKernel(0.05)] }), (D) => D + 2],
  ['WhiteKernel + Matérn', (m) => new SumKernel({ kernels: [new WhiteKernel(0.05), m] }), (D) => D + 2],
];

describe.each([1.5, 2.5, Infinity])('Matérn ν=%s analytic gradient', (nu) => {
  describe.each(COMPOSITIONS)('%s', (_name, compose, nSlots) => {
    // A scalar alpha and a heteroscedastic one both sit on the diagonal
    // untouched by the optimizer; the gradient must be right under either.
    it.each([
      ['scalar alpha', () => 5e-2],
      ['per-observation alpha', (n) => Array.from({ length: n }, (_, i) => 0.02 + 0.001 * i)],
    ])('matches finite differences of the negative log marginal likelihood (%s)', (_a, alphaFor) => {
      const D = 5;
      const { X, y } = makeData(42, 90, D);
      const kernel = compose(new Matern({ lengthScale: Array(D).fill(1.3), nu }));
      const gp = new GaussianProcessRegressor({ kernel, alpha: alphaFor(X.length), optimize: false, normalizeY: true });
      gp.fit(X, y);
      const hs = hyperList(gp);
      const logv = hs.map((h) => Math.log(h.get()));

      const analytic = gp._negLogMLGrad(logv.slice(), hs).gradient;
      const nll = (lv) => { hs.forEach((h, i) => h.set(Math.max(h.min, Math.exp(lv[i])))); return gp._negLogML(); };
      const h = 1e-5;
      const fd = logv.map((_, i) => {
        const a = logv.slice(); a[i] += h;
        const b = logv.slice(); b[i] -= h;
        return (nll(a) - nll(b)) / (2 * h);
      });

      expect(analytic.length).toBe(nSlots(D));
      expect(hs.length).toBe(nSlots(D));
      for (let i = 0; i < analytic.length; i++) expect(analytic[i]).toBeCloseTo(fd[i], 4);
    });
  });

  it(`ν=${nu}: a Matérn + WhiteKernel stays on the analytic path`, () => {
    const { X, y } = makeData(42, 30, 2);
    const gp = new GaussianProcessRegressor({
      kernel: new SumKernel({ kernels: [new Matern({ lengthScale: [1, 1], nu }), new WhiteKernel(0.05)] }),
    });
    gp.fit(X, y);
    expect(gp._analyticParts()).not.toBeNull();
  });
});

describe('Matérn GP optimize (analytic path)', () => {
  it('improves the log marginal likelihood over the un-optimized start', () => {
    const D = 6;
    const { X, y } = makeData(7, 100, D);
    const start = new GaussianProcessRegressor({ kernel: new Matern({ lengthScale: Array(D).fill(1), nu: 2.5 }), alpha: 1e-2, optimize: false, normalizeY: true });
    start.fit(X, y);
    const opt = new GaussianProcessRegressor({ kernel: new Matern({ lengthScale: Array(D).fill(1), nu: 2.5 }), alpha: 1e-2, optimize: true, nRestarts: 0, randomState: 42, normalizeY: true });
    opt.fit(X, y);
    expect(opt.logMarginalLikelihood()).toBeGreaterThan(start.logMarginalLikelihood());
  });

  it('recovers a relevant vs irrelevant ARD length scale', () => {
    // y depends on x0 only; x1..x4 are noise. The x0 length scale should end up
    // markedly shorter (more relevant) than the noise dimensions.
    const { X, y } = makeData(3, 120, 5);
    const gp = new GaussianProcessRegressor({ kernel: new Matern({ lengthScale: Array(5).fill(1), nu: 2.5 }), alpha: 1e-2, optimize: true, nRestarts: 0, randomState: 1, normalizeY: true });
    gp.fit(X, y);
    const ls = gp.kernel.lengthScale;
    const noiseMedian = [ls[3], ls[4]].sort((a, b) => a - b)[0];
    expect(ls[0]).toBeLessThan(noiseMedian);
  });
});
