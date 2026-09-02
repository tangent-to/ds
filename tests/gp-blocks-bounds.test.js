/**
 * ARD by block, and bounds on hyperparameters.
 *
 * Both exist for the same reason: marginal-likelihood tuning of one length
 * scale per feature, on a few hundred rows, is under-constrained. Measured on
 * a 340-row agronomic model with 37 features and a learned noise, the noise
 * collapsed to zero in four folds of seven. Fewer length scales, or a floor
 * on the noise, are the two ways to stop that from inside the model.
 */

import { describe, expect, it } from 'vitest';
import { variable, toNested } from '@tangent.to/grad';
import { GaussianProcessRegressor, Matern, RBF, SumKernel, WhiteKernel } from '../src/ml/index.js';
import { kernelConstants, kernelMatrixAD } from '../src/ml/kernels/autodiff.js';

const X = [[0.3, -1.2, 0.5], [1.1, 0.4, -0.2], [-0.7, 2.0, 1.1], [2.2, 0.1, 0.0], [0.0, 0.0, 0.7], [0.9, -0.4, -1.3]];
const y = X.map(([a, b, c]) => Math.sin(a) + 0.4 * b - 0.3 * c * c);

describe('ARD by block', () => {
  it('is the full-ARD kernel with the block scales repeated', () => {
    const blocked = new Matern({ lengthScale: [1.3, 0.6], blocks: [0, 0, 1], nu: 2.5, amplitude: 1.1 });
    const full = new Matern({ lengthScale: [1.3, 1.3, 0.6], nu: 2.5, amplitude: 1.1 });
    for (const a of X) for (const b of X) expect(blocked.compute(a, b)).toBeCloseTo(full.compute(a, b), 14);
    const rbfB = new RBF({ lengthScale: [1.3, 0.6], blocks: [0, 0, 1], variance: 0.8 });
    const rbfF = new RBF({ lengthScale: [1.3, 1.3, 0.6], variance: 0.8 });
    for (const a of X) for (const b of X) expect(rbfB.compute(a, b)).toBeCloseTo(rbfF.compute(a, b), 14);
  });

  it('builds the same matrix on the autodiff path, from two hyperparameters not three', () => {
    const consts = kernelConstants(X);
    const k = new Matern({ lengthScale: [1.3, 0.6], blocks: [0, 0, 1], nu: 2.5, amplitude: 1.1 });
    const ref = k.call(X).to2DArray();
    const { K, next } = kernelMatrixAD(k, consts, variable([1.3, 0.6, 1.1]), 0);
    expect(next).toBe(3); // two scales and the variance
    const got = toNested(K.value);
    for (let i = 0; i < X.length; i++) for (let j = 0; j < X.length; j++) expect(got[i][j]).toBeCloseTo(ref[i][j], 12);
  });

  it('differentiates the likelihood in the block scales, checked against finite differences', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1.2, 0.7], blocks: [0, 0, 1], nu: 2.5, amplitude: 0.9 }), alpha: 0.1,
    });
    gp.fit(X, y);
    const k = gp.kernel;
    const hs = [
      { get: () => k.lengthScale[0], set: (v) => { k.lengthScale[0] = v; }, min: 1e-5 },
      { get: () => k.lengthScale[1], set: (v) => { k.lengthScale[1] = v; }, min: 1e-5 },
      { get: () => k.variance, set: (v) => { k.variance = v; }, min: 1e-8 },
    ];
    const consts = kernelConstants(X);
    const logv = hs.map((h) => Math.log(h.get()));
    const analytic = gp._negLogMLGradAD(logv.slice(), hs, consts).gradient;
    const nll = (lv) => { hs.forEach((h, i) => h.set(Math.exp(lv[i]))); return gp._negLogML(); };
    const h = 1e-5;
    logv.forEach((_, i) => {
      const a = logv.slice(); a[i] += h;
      const b = logv.slice(); b[i] -= h;
      expect(analytic[i]).toBeCloseTo((nll(a) - nll(b)) / (2 * h), 5);
    });
  });

  it('tunes two scales under optimize: true, moving them, and keeps them shared', () => {
    // A blocked Matérn must take the autodiff gradient, not the hand-derived
    // one, which knows nothing of blocks: it indexed lengthScale by
    // dimension, read undefined past the block count, and handed the
    // optimizer NaN, which stopped at the initial values without a word.
    // So the assertion is that the scales MOVED and the likelihood rose.
    const before = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1, 1], blocks: [0, 0, 1], nu: 2.5 }), alpha: 0.05,
    });
    before.fit(X, y);
    const nllBefore = before._negLogML();
    const gp = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1, 1], blocks: [0, 0, 1], nu: 2.5 }), alpha: 0.05, optimize: true, nRestarts: 0,
    });
    gp.fit(X, y);
    expect(gp.kernel.lengthScale).toHaveLength(2);
    expect(gp.kernel.blocks).toEqual([0, 0, 1]);
    expect(gp.kernel.lengthScale.every((l) => l > 0 && Number.isFinite(l))).toBe(true);
    expect(gp.kernel.lengthScale).not.toEqual([1, 1]);
    expect(gp._negLogML()).toBeLessThan(nllBefore);
  });

  it('with a WhiteKernel in the sum, still moves', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new SumKernel({ kernels: [new Matern({ lengthScale: [1, 1], blocks: [0, 0, 1], nu: 2.5 }), new WhiteKernel(0.1)] }),
      alpha: 1e-10, optimize: true, nRestarts: 0,
    });
    gp.fit(X, y);
    expect(gp.kernel.kernels[0].lengthScale).not.toEqual([1, 1]);
  });

  it('survives getParams and a rebuild', () => {
    const k = new Matern({ lengthScale: [1.3, 0.6], blocks: [0, 0, 1], nu: 2.5, lengthScaleBounds: [0.1, 10] });
    const again = new Matern(k.getParams());
    expect(again.blocks).toEqual([0, 0, 1]);
    expect(again.lengthScaleBounds).toEqual([0.1, 10]);
    expect(again.compute(X[0], X[1])).toBe(k.compute(X[0], X[1]));
  });

  it('refuses a block map that does not fit the length scales', () => {
    expect(() => new Matern({ lengthScale: 1, blocks: [0, 0, 1] })).toThrow(/needs an array lengthScale/);
    expect(() => new Matern({ lengthScale: [1, 1], blocks: [0, 0, 2] })).toThrow(/indices into lengthScale \(0 to 1\)/);
    expect(() => new Matern({ lengthScale: [1, 1, 1], blocks: [0, 0, 1] })).toThrow(/3 entries but blocks uses 2/);
  });
});

describe('hyperparameter bounds', () => {
  it('a floor on the noise holds under optimization', () => {
    // Nearly noiseless data, where unbounded marginal likelihood drives the
    // noise toward its 1e-10 default floor. With a floor of 0.05 it stops there.
    const yClean = X.map(([a, b]) => a + 0.5 * b);
    const gp = new GaussianProcessRegressor({
      kernel: new SumKernel({ kernels: [new RBF(1, 1), new WhiteKernel({ noiseLevel: 0.5, noiseLevelBounds: [0.05, 2] })] }),
      alpha: 1e-10, optimize: true, nRestarts: 0,
    });
    gp.fit(X, yClean);
    const noise = gp.kernel.kernels[1].noiseLevel;
    expect(noise).toBeGreaterThanOrEqual(0.05 - 1e-12);
    expect(noise).toBeLessThanOrEqual(2 + 1e-12);
  });

  it('a ceiling on a length scale holds too', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new RBF({ lengthScale: 1, variance: 1, lengthScaleBounds: [0.2, 1.5] }), alpha: 0.1, optimize: true, nRestarts: 0,
    });
    gp.fit(X, y);
    expect(gp.kernel.lengthScale).toBeGreaterThanOrEqual(0.2 - 1e-12);
    expect(gp.kernel.lengthScale).toBeLessThanOrEqual(1.5 + 1e-12);
  });

  it('rejects malformed bounds at fit time', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new WhiteKernel({ noiseLevel: 0.1, noiseLevelBounds: [1, 0.5] }), optimize: true,
    });
    expect(() => gp.fit(X, y)).toThrow(/bounds must be \[low, high\]/);
  });
});
