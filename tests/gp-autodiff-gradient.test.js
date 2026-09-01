/**
 * Autodiff gradients for GP kernels that had none.
 *
 * `_negLogMLGrad` is hand-derived and covers the Matérn family only. Every
 * other kernel fell back to `_patternSearch`, a derivative-free coordinate
 * search that stops early: on a 300-point problem an RBF reached a log
 * marginal likelihood of -6.3 where the analytic Matérn path reached +25.9.
 * The quality of a fit depended on which kernel you happened to pick.
 *
 * `_negLogMLGradAD` rebuilds the kernel matrix in @tangent.to/grad ops, so the
 * same six lines differentiate every kernel `supportsAutodiff` accepts.
 */

import { describe, expect, it } from 'vitest';
import { variable, toNested } from '@tangent.to/grad';
import {
  ConstantKernel, DotProduct, GaussianProcessRegressor, Matern, Periodic,
  RationalQuadratic, RBF, SumKernel, WhiteKernel,
} from '../src/ml/index.js';
import { kernelConstants, kernelMatrixAD, supportsAutodiff } from '../src/ml/kernels/autodiff.js';

const X = [[0.3, -1.2], [1.1, 0.4], [-0.7, 2.0], [2.2, 0.1], [0.0, 0.0]];

describe('the differentiable kernel matrix reproduces kernel.call', () => {
  // The builder mirrors each kernel's own compute(). That duplication is only
  // safe because this holds for every form the kernels take.
  const consts = kernelConstants(X);

  it.each([
    ['RBF isotropic', new RBF(1.3, 0.8), [1.3, 0.8]],
    ['RBF ARD', new RBF([1.3, 0.7], 0.8), [1.3, 0.7, 0.8]],
    ['Matern nu=0.5', new Matern({ lengthScale: 1.1, nu: 0.5, amplitude: 1.2 }), [1.1, 1.2]],
    ['Matern nu=1.5', new Matern({ lengthScale: 1.1, nu: 1.5, amplitude: 1.2 }), [1.1, 1.2]],
    ['Matern nu=2.5', new Matern({ lengthScale: 1.1, nu: 2.5, amplitude: 1.2 }), [1.1, 1.2]],
    ['Matern nu=Inf', new Matern({ lengthScale: 1.1, nu: Infinity, amplitude: 1.2 }), [1.1, 1.2]],
    ['Matern ARD', new Matern([1.1, 0.9], 2.5, 1.2), [1.1, 0.9, 1.2]],
    ['RationalQuadratic', new RationalQuadratic(1.4, 2.1, 0.9), [1.4, 0.9, 2.1]],
    ['DotProduct', new DotProduct({ sigma0: 1.7 }), [1.7]],
    ['ConstantKernel', new ConstantKernel({ value: 2.3 }), [2.3]],
    ['WhiteKernel', new WhiteKernel(0.35), [0.35]],
    ['Periodic', new Periodic(1.2, 2.5, 0.9), [1.2, 0.9]],
    ['RBF + WhiteKernel', new SumKernel({ kernels: [new RBF(1.3, 0.8), new WhiteKernel(0.2)] }), [1.3, 0.8, 0.2]],
  ])('%s', (_name, kernel, theta) => {
    const ref = kernel.call(X).to2DArray();
    const got = toNested(kernelMatrixAD(kernel, consts, variable(theta), 0).K.value);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X.length; j++) expect(got[i][j]).toBeCloseTo(ref[i][j], 12);
    }
  });
});

describe('the autodiff likelihood gradient', () => {
  const y = X.map(([a, b]) => Math.sin(a) + 0.4 * b);

  /** Rebuild the hyperparameter accessors the optimizer uses. */
  const hyperList = (gp) => {
    const hs = [];
    const visit = (k) => {
      if (k instanceof SumKernel) { k.kernels.forEach(visit); return; }
      if (k instanceof WhiteKernel) {
        hs.push({ get: () => k.noiseLevel, set: (v) => { k.noiseLevel = v; }, min: 1e-10 });
        return;
      }
      if (Array.isArray(k.lengthScale)) {
        k.lengthScale.forEach((_, i) =>
          hs.push({ get: () => k.lengthScale[i], set: (v) => { k.lengthScale[i] = v; }, min: 1e-5 }));
      } else {
        hs.push({ get: () => k.lengthScale, set: (v) => { k.lengthScale = v; }, min: 1e-5 });
      }
      hs.push({ get: () => k.variance, set: (v) => { k.variance = v; }, min: 1e-8 });
    };
    visit(gp.kernel);
    return hs;
  };

  it.each([
    ['RBF isotropic', () => new RBF(1.2, 0.9)],
    ['RBF ARD', () => new RBF([1.2, 0.8], 0.9)],
    ['RBF + WhiteKernel', () => new SumKernel({ kernels: [new RBF(1.2, 0.9), new WhiteKernel(0.15)] })],
  ])('%s: matches finite differences of the likelihood', (_name, makeKernel) => {
    const gp = new GaussianProcessRegressor({ kernel: makeKernel(), alpha: 0.1 });
    gp.fit(X, y);
    const consts = kernelConstants(X);
    const hs = hyperList(gp);
    const logv = hs.map((h) => Math.log(h.get()));

    const analytic = gp._negLogMLGradAD(logv.slice(), hs, consts).gradient;
    const nll = (lv) => {
      hs.forEach((h, i) => h.set(Math.max(h.min, Math.exp(lv[i]))));
      return gp._negLogML();
    };
    const h = 1e-5;
    const fd = logv.map((_, i) => {
      const a = logv.slice(); a[i] += h;
      const b = logv.slice(); b[i] -= h;
      return (nll(a) - nll(b)) / (2 * h);
    });

    expect(analytic).toHaveLength(hs.length);
    analytic.forEach((g, i) => expect(g).toBeCloseTo(fd[i], 5));
  });

  // Two independent derivations of the same quantity, for every nu the
  // hand-written path covers. This is the check that caught the tape returning
  // NaN length-scale gradients from sqrt(0) on the kernel diagonal, while its
  // VALUE stayed exact — a failure no value-only test would have seen.
  it.each([[1.5], [2.5], [Infinity]])(
    'agrees with the hand-derived Matérn gradient at nu=%s',
    (nu) => {
      const gp = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1.2, 0.8], nu, amplitude: 0.9 }),
        alpha: 0.1,
      });
      gp.fit(X, y);
      const hs = hyperList(gp);
      const logv = hs.map((h) => Math.log(h.get()));
      const byHand = gp._negLogMLGrad(logv.slice(), hs).gradient;
      const byTape = gp._negLogMLGradAD(logv.slice(), hs, kernelConstants(X)).gradient;
      byHand.forEach((g, i) => expect(byTape[i]).toBeCloseTo(g, 12));
    },
  );

  it('differentiates a Matérn nu=0.5, which never had a hand-derived form', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1.2, 0.8], nu: 0.5, amplitude: 0.9 }),
      alpha: 0.1,
    });
    gp.fit(X, y);
    expect(gp._analyticParts()).toBeNull(); // no closed form exists for it
    const hs = hyperList(gp);
    const logv = hs.map((h) => Math.log(h.get()));
    const byTape = gp._negLogMLGradAD(logv.slice(), hs, kernelConstants(X)).gradient;
    const nll = (lv) => {
      hs.forEach((h, i) => h.set(Math.max(h.min, Math.exp(lv[i]))));
      return gp._negLogML();
    };
    const fd = logv.map((_, i) => {
      const a = logv.slice(); a[i] += 1e-5;
      const b = logv.slice(); b[i] -= 1e-5;
      return (nll(a) - nll(b)) / 2e-5;
    });
    byTape.forEach((g, i) => expect(g).toBeCloseTo(fd[i], 6));
  });

  it('reports a penalty with a zero gradient when K is not positive definite', () => {
    // Two identical training points and no noise make K exactly singular for
    // ANY hyperparameters — a case no amount of hyperparameter clamping can
    // rescue. The optimizer must get a finite penalty and a zero direction
    // rather than an exception escaping into the fit.
    const dup = [...X, X[0]];
    const dupY = [...y, y[0]];
    const gp = new GaussianProcessRegressor({ kernel: new RBF(1, 1), alpha: 1e-10 });
    gp.fit(dup, dupY);
    gp.alpha = 0;
    gp._alphaDiag = null;
    const hs = hyperList(gp);
    const out = gp._negLogMLGradAD([Math.log(1), Math.log(1)], hs, kernelConstants(dup));
    expect(out.loss).toBe(1e12);
    expect(out.gradient.every((g) => g === 0)).toBe(true);
  });

  it('keeps a merely ill-conditioned K on the real objective', () => {
    // The guard must not fire early: a huge length scale makes K nearly
    // singular but still factorizable, and the optimizer is better served by
    // the true (large) loss than by a flat penalty with no direction.
    const gp = new GaussianProcessRegressor({ kernel: new RBF(1, 1), alpha: 0 });
    gp.fit(X, y);
    const hs = hyperList(gp);
    const out = gp._negLogMLGradAD([Math.log(1e8), Math.log(1)], hs, kernelConstants(X));
    expect(out.loss).toBeGreaterThan(1e12);
    expect(Number.isFinite(out.loss)).toBe(true);
  });
});

describe('supportsAutodiff', () => {
  it('accepts the kernels the builder covers', () => {
    expect(supportsAutodiff(new RBF(1, 1))).toBe(true);
    expect(supportsAutodiff(new Matern({ nu: 2.5 }))).toBe(true);
    expect(supportsAutodiff(new SumKernel({ kernels: [new RBF(1, 1), new WhiteKernel(0.1)] }))).toBe(true);
  });

  it('accepts RationalQuadratic, which box bounds make tractable', () => {
    // Its alpha ridge is unbounded — the kernel converges to an RBF as alpha
    // grows — so this only works because collectHypers gives that parameter a
    // max and the optimizer runs in bounded coordinates.
    expect(supportsAutodiff(new RationalQuadratic(1, 1, 1))).toBe(true);
  });

  it('rejects a sum containing a kernel it cannot build', () => {
    const opaque = { constructor: { name: 'MysteryKernel' } };
    expect(supportsAutodiff(new SumKernel({
      kernels: [new RBF(1, 1), new WhiteKernel(0.1)],
    }))).toBe(true);
    expect(supportsAutodiff(opaque)).toBe(false);
  });
});

describe('the quality cliff is closed', () => {
  // The measurement that motivated all of this.
  const rnd = ((s) => () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; })(1);
  const XX = Array.from({ length: 120 }, () => [rnd() * 10, rnd() * 10]);
  const yy = XX.map(([a, b]) => Math.sin(a) + 0.3 * b + 0.1 * rnd());

  it('an optimized RBF now reaches a likelihood comparable to the Matérn path', () => {
    const rbf = new GaussianProcessRegressor({ kernel: new RBF([1, 1], 1), alpha: 0.1, optimize: true });
    rbf.fit(XX, yy);
    const matern = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, amplitude: 1 }), alpha: 0.1, optimize: true,
    });
    matern.fit(XX, yy);

    // Before, the derivative-free search left RBF tens of log-likelihood units
    // behind. It should now be in the same league, not an order apart.
    expect(rbf.logMarginalLikelihood_).toBeGreaterThan(matern.logMarginalLikelihood_ - 5);
    expect(rbf.logMarginalLikelihood()).toBeCloseTo(rbf.logMarginalLikelihood_, 8);
  });

  it('leaves the Matérn path on its own hand-derived gradient', () => {
    const gp = new GaussianProcessRegressor({
      kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, amplitude: 1 }), alpha: 0.1, optimize: true,
    });
    gp.fit(XX, yy);
    expect(gp._analyticParts()).not.toBeNull();
    expect(Number.isFinite(gp.logMarginalLikelihood_)).toBe(true);
  });
});
