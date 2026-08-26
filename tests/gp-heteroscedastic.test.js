/**
 * Heteroscedastic observation noise for the Gaussian Process regressor.
 *
 * `alpha` may be a per-observation array of noise variances instead of one
 * scalar, as in sklearn — the case for measurements of unequal reliability
 * (polls weighted by sample size, sensors with varying measurement error).
 *
 * The sklearn reference numbers below come from `src/ml/test-gp-vs-sklearn.py`
 * ("Test 8: Heteroscedastic alpha"); re-run that script to regenerate them.
 * They are inlined rather than produced live so the test runs without a Python
 * toolchain, matching scripts/compare-gp-vs-sklearn.js.
 */

import { describe, expect, it } from "vitest";
import { GaussianProcessRegressor } from "../src/ml/estimators/GaussianProcessRegressor.js";
import { Matern, RBF, SumKernel, WhiteKernel } from "../src/ml/kernels/index.js";

const X = [[0], [1], [2], [3], [4], [5]];
const y = [0.0, 0.8, 0.2, 1.4, 0.1, 1.1];
// Point 2 is nearly noise-free, point 4 is barely trusted at all.
const ALPHA = [0.01, 0.5, 0.001, 0.2, 2.0, 0.05];
const X_TEST = [[0], [0.5], [1], [2], [2.5], [4], [5]];

// sklearn: GaussianProcessRegressor(kernel=RBF(1.5), alpha=ALPHA, optimizer=None)
const SKLEARN_MEAN = [
  0.004382141814764796, -0.026103192652084578, -0.04168813806834361,
  0.20310471784568768, 0.49940199545151054, 1.227113152306242,
  1.0639289915961496,
];
const SKLEARN_STD = [
  0.09912178275209248, 0.19767340106784612, 0.24114627387768117,
  0.031559700056692716, 0.17645053205823966, 0.3802981501034292,
  0.21618048654210603,
];
const SKLEARN_LML = -7.841782301377197;

const newGP = (opts = {}) =>
  new GaussianProcessRegressor({ kernel: new RBF(1.5, 1.0), ...opts });

describe("GP heteroscedastic alpha", () => {
  describe("matches sklearn with an array-valued alpha", () => {
    it("reproduces sklearn's posterior mean and std", () => {
      const gp = newGP().fit(X, y, { alpha: ALPHA });
      const { mean, std } = gp.predict(X_TEST, { returnStd: true });

      mean.forEach((m, i) => expect(m).toBeCloseTo(SKLEARN_MEAN[i], 10));
      std.forEach((s, i) => expect(s).toBeCloseTo(SKLEARN_STD[i], 10));
    });

    it("reproduces sklearn's log marginal likelihood", () => {
      const gp = newGP().fit(X, y, { alpha: ALPHA });
      expect(gp.logMarginalLikelihood_).toBeCloseTo(SKLEARN_LML, 10);
      expect(gp.logMarginalLikelihood()).toBeCloseTo(SKLEARN_LML, 10);
    });

    it("accepts the noise vector via the constructor as well as fit()", () => {
      const viaCtor = newGP({ alpha: ALPHA }).fit(X, y);
      const viaFit = newGP().fit(X, y, { alpha: ALPHA });
      expect(viaCtor.predict(X_TEST)).toEqual(viaFit.predict(X_TEST));
    });

    it("accepts a typed array", () => {
      const gp = newGP().fit(X, y, { alpha: Float64Array.from(ALPHA) });
      gp.predict(X_TEST).forEach((m, i) => expect(m).toBeCloseTo(SKLEARN_MEAN[i], 10));
    });
  });

  describe("noise weighting", () => {
    it("tracks the trusted point and lets the noisy one go", () => {
      const gp = newGP().fit(X, y, { alpha: ALPHA });
      const fitted = gp.predict(X);
      // x=2 has alpha 0.001 -> the posterior is pinned to y=0.2 there.
      expect(fitted[2]).toBeCloseTo(0.2, 2);
      // x=4 has alpha 2.0 -> y=0.1 is treated as almost uninformative, so the
      // posterior follows the neighbours upward instead.
      expect(Math.abs(fitted[4] - 0.1)).toBeGreaterThan(1.0);
    });

    it("gives the low-noise points the tighter predictive interval", () => {
      const { std } = newGP().fit(X, y, { alpha: ALPHA }).predict(X, { returnStd: true });
      expect(std[2]).toBeLessThan(std[4]); // alpha 0.001 vs 2.0
      expect(std[0]).toBeLessThan(std[1]); // alpha 0.01  vs 0.5
    });

    it("weights an observation the same as replicating a scalar-noise fit", () => {
      // Down-weighting one point is not the same as deleting it, but pushing
      // its variance to ~infinity is: the fit must converge on the 5-point fit.
      const muted = ALPHA.map((a, i) => (i === 4 ? 1e8 : a));
      const dropped = [0, 1, 2, 3, 5];
      const full = newGP().fit(X, y, { alpha: muted }).predict(X_TEST);
      const subset = newGP().fit(
        dropped.map((i) => X[i]),
        dropped.map((i) => y[i]),
        { alpha: dropped.map((i) => ALPHA[i]) },
      ).predict(X_TEST);
      full.forEach((m, i) => expect(m).toBeCloseTo(subset[i], 6));
    });
  });

  describe("backward compatibility", () => {
    it("leaves the scalar path untouched", () => {
      const gp = newGP({ alpha: 0.1 }).fit(X, y);
      const { mean, std } = gp.predict(X_TEST, { returnStd: true });
      expect(mean.every(Number.isFinite)).toBe(true);
      expect(std.every((s) => s > 0)).toBe(true);
      expect(gp.alpha).toBe(0.1);
      expect(gp._alphaDiag).toBeNull();
    });

    it("makes a constant array identical to the equivalent scalar", () => {
      const scalar = newGP({ alpha: 0.1 }).fit(X, y);
      const vector = newGP().fit(X, y, { alpha: new Array(6).fill(0.1) });
      const a = scalar.predict(X_TEST, { returnStd: true });
      const b = vector.predict(X_TEST, { returnStd: true });
      a.mean.forEach((m, i) => expect(m).toBeCloseTo(b.mean[i], 12));
      a.std.forEach((s, i) => expect(s).toBeCloseTo(b.std[i], 12));
      expect(scalar.logMarginalLikelihood_).toBeCloseTo(vector.logMarginalLikelihood_, 12);
    });

    it("round-trips the noise vector through toJSON/fromJSON", () => {
      const gp = newGP().fit(X, y, { alpha: ALPHA });
      const revived = GaussianProcessRegressor.fromJSON(JSON.parse(JSON.stringify(gp)));
      expect(revived.alpha).toEqual(ALPHA);
      expect(revived._alphaDiag).toEqual(ALPHA);
      revived.predict(X_TEST).forEach((m, i) => expect(m).toBeCloseTo(SKLEARN_MEAN[i], 10));
    });
  });

  describe("validation", () => {
    it("rejects a noise vector whose length does not match y", () => {
      expect(() => newGP().fit(X, y, { alpha: [0.1, 0.2] })).toThrow(/length 2 but there are 6/);
    });

    it("rejects negative or non-finite variances", () => {
      const bad = ALPHA.slice();
      bad[3] = -1;
      expect(() => newGP().fit(X, y, { alpha: bad })).toThrow(/alpha\[3\]/);
      bad[3] = NaN;
      expect(() => newGP().fit(X, y, { alpha: bad })).toThrow(/alpha\[3\]/);
    });

    it("rejects a non-numeric scalar", () => {
      expect(() => newGP().fit(X, y, { alpha: "0.1" })).toThrow(/finite number or an array/);
    });
  });

  describe("hyperparameter optimization stays consistent with the fit noise", () => {
    // The optimizer must score candidate kernels under the SAME K + noise that
    // _refit() finally factorizes. Tuning under a scalar while fitting under a
    // vector silently optimizes the wrong objective.
    const cases = [
      ["derivative-free (RBF)", () => new RBF(1.0, 1.0)],
      ["analytic gradients (Matérn)", () => new Matern({ lengthScale: 1.0, nu: 2.5, amplitude: 1.0 })],
    ];

    for (const [name, kernel] of cases) {
      it(`${name}: the reported likelihood is the one being maximized`, () => {
        const gp = new GaussianProcessRegressor({ kernel: kernel(), optimize: true });
        gp.fit(X, y, { alpha: ALPHA });
        // logMarginalLikelihood_ is read off the fit factorization;
        // logMarginalLikelihood() rebuilds K + noise from scratch. They agree
        // only if both use the vector.
        expect(gp.logMarginalLikelihood()).toBeCloseTo(gp.logMarginalLikelihood_, 10);
      });

      it(`${name}: the tuned kernel beats its neighbours under the vector noise`, () => {
        const gp = new GaussianProcessRegressor({ kernel: kernel(), optimize: true });
        gp.fit(X, y, { alpha: ALPHA });
        const best = gp.logMarginalLikelihood_;

        for (const factor of [0.5, 2.0]) {
          const probe = new GaussianProcessRegressor({ kernel: kernel(), alpha: ALPHA });
          probe.kernel.lengthScale = gp.kernel.lengthScale * factor;
          probe.kernel.variance = gp.kernel.variance;
          probe.fit(X, y);
          expect(probe.logMarginalLikelihood_).toBeLessThan(best + 1e-6);
        }
      });

      it(`${name}: leaves the supplied noise variances untouched`, () => {
        const gp = new GaussianProcessRegressor({ kernel: kernel(), optimize: true });
        gp.fit(X, y, { alpha: ALPHA.slice() });
        expect(gp.alpha).toEqual(ALPHA);
        expect(gp._alphaDiag).toEqual(ALPHA);
      });
    }

    it("never tunes alpha, not even a scalar one", () => {
      // `alpha` is noise the caller declares as known. Learning a noise level
      // is what WhiteKernel is for.
      const gp = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: 1.0, nu: 2.5, amplitude: 1.0 }),
        alpha: 0.5,
        optimize: true,
      });
      gp.fit(X, y);
      expect(gp.alpha).toBe(0.5);
      expect(gp.logMarginalLikelihood()).toBeCloseTo(gp.logMarginalLikelihood_, 10);
    });

    it("tunes a WhiteKernel's noiseLevel instead, alongside a fixed alpha", () => {
      const alpha = ALPHA.slice();
      const white = new WhiteKernel(0.5);
      const gp = new GaussianProcessRegressor({
        kernel: new SumKernel({
          kernels: [new Matern({ lengthScale: 1.0, nu: 2.5, amplitude: 1.0 }), white],
        }),
        optimize: true,
      });
      gp.fit(X, y, { alpha });
      expect(white.noiseLevel).not.toBe(0.5); // learned
      expect(white.noiseLevel).toBeGreaterThan(0);
      expect(gp.alpha).toEqual(ALPHA); // declared, untouched
      expect(gp.logMarginalLikelihood()).toBeCloseTo(gp.logMarginalLikelihood_, 10);
    });
  });
});

describe("GP heteroscedastic alpha - typed array serialization", () => {
  it("round-trips a typed-array noise vector as a plain array", () => {
    const gp = new GaussianProcessRegressor({ kernel: new RBF(1.5, 1.0) })
      .fit(X, y, { alpha: Float64Array.from(ALPHA) });
    const revived = GaussianProcessRegressor.fromJSON(JSON.parse(JSON.stringify(gp)));
    expect(revived.alpha).toEqual(ALPHA);
    expect(revived._alphaDiag).toEqual(ALPHA);
  });
});
