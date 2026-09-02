/**
 * `predictGradient`: the predictive mean and standard deviation at one
 * input, with their gradients in that input.
 *
 * Two references. The values must equal `predict`'s, to rounding, since they
 * are the same quantities through a different path. The gradients must match
 * central finite differences of `predict`, which is the check that the
 * differentiable cross kernel is the kernel.
 */

import { describe, expect, it } from 'vitest';
import { GaussianProcessRegressor, Matern, RBF, SumKernel, WhiteKernel, ConstantKernel } from '../src/ml/index.js';

const X = [[0.3, -1.2, 0.5], [1.1, 0.4, -0.2], [-0.7, 2.0, 1.1], [2.2, 0.1, 0.0], [0.0, 0.0, 0.7], [0.9, -0.4, -1.3], [-1.4, 0.8, 0.3]];
const y = X.map(([a, b, c]) => Math.sin(a) + 0.4 * b - 0.3 * c * c);
const at = [0.4, 0.2, -0.1];

function fd(gp, x, h = 1e-6) {
  const dm = [], ds = [];
  for (let k = 0; k < x.length; k++) {
    const a = x.slice(); a[k] += h;
    const b = x.slice(); b[k] -= h;
    const pa = gp.predict([a], { returnStd: true });
    const pb = gp.predict([b], { returnStd: true });
    dm.push((pa.mean[0] - pb.mean[0]) / (2 * h));
    ds.push((pa.std[0] - pb.std[0]) / (2 * h));
  }
  return { dm, ds };
}

describe('predictGradient', () => {
  const kernels = {
    'RBF isotropic': () => new RBF(1.1, 0.9),
    'RBF ARD': () => new RBF([1.1, 0.7, 1.4], 0.9),
    'RBF by block': () => new RBF({ lengthScale: [1.1, 0.7], blocks: [0, 0, 1], variance: 0.9 }),
    'Matern 1.5': () => new Matern({ lengthScale: 1.2, nu: 1.5, amplitude: 1.1 }),
    'Matern 2.5 ARD': () => new Matern({ lengthScale: [1.2, 0.8, 1.5], nu: 2.5, amplitude: 1.1 }),
    'Matern 0.5': () => new Matern({ lengthScale: 1.2, nu: 0.5 }),
    'Matern + White + Constant': () => new SumKernel({ kernels: [
      new Matern({ lengthScale: [1.2, 0.8, 1.5], nu: 2.5 }), new WhiteKernel(0.2), new ConstantKernel({ value: 0.3 }),
    ] }),
  };

  for (const [name, make] of Object.entries(kernels)) {
    it(`${name}: values equal predict, gradients match finite differences`, () => {
      const gp = new GaussianProcessRegressor({ kernel: make(), alpha: 0.05, normalizeY: true });
      gp.fit(X, y);
      const ref = gp.predict([at], { returnStd: true });
      const g = gp.predictGradient(at);
      expect(g.mean).toBeCloseTo(ref.mean[0], 10);
      expect(g.std).toBeCloseTo(ref.std[0], 10);
      const { dm, ds } = fd(gp, at);
      g.meanGradient.forEach((v, k) => expect(v, `dmean/dx${k}`).toBeCloseTo(dm[k], 5));
      g.stdGradient.forEach((v, k) => expect(v, `dstd/dx${k}`).toBeCloseTo(ds[k], 5));
    });
  }

  it('is compiled once per fit and replayed', () => {
    const gp = new GaussianProcessRegressor({ kernel: new RBF([1.1, 0.7, 1.4], 0.9), alpha: 0.05 });
    gp.fit(X, y);
    const a = gp.predictGradient(at);
    const plan = gp._predictAD;
    const b = gp.predictGradient([0.1, 0.1, 0.1]);
    expect(gp._predictAD).toBe(plan);
    expect(a.mean).not.toBe(b.mean);
    const shifted = y.map((v) => v + 1);
    gp.fit(X, shifted);
    expect(gp._predictAD).toBeNull(); // a refit discards the plan
    expect(gp.predictGradient(at).mean).toBeCloseTo(gp.predict([at])[0], 10); // and rebuilds it on the new fit
  });

  it('refuses a wrong-length input and an unfitted model', () => {
    const gp = new GaussianProcessRegressor({ kernel: new RBF(1, 1) });
    expect(() => gp.predictGradient(at)).toThrow(/fit/);
    gp.fit(X, y);
    expect(() => gp.predictGradient([1, 2])).toThrow(/expected one input of length 3/);
  });
});
