import { describe, expect, it } from "vitest";
import { GaussianProcessRegressor } from "../src/ml/estimators/GaussianProcessRegressor.js";
import { Matern, RBF, DotProduct, SumKernel } from "../src/ml/kernels/index.js";

// Deterministic 1-D PRNG for synthetic data.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("GP kernel fitting", () => {
  describe("ARD Matérn", () => {
    it("accepts a per-dimension length-scale array and stays well-defined", () => {
      const k = new Matern({ lengthScale: [1.0, 2.0], nu: 2.5, variance: 1.0 });
      expect(k.compute([0, 0], [0, 0])).toBeCloseTo(1.0, 12); // variance at r=0
      // Larger length scale on dim 1 => a unit step there is "closer".
      const c0 = k.compute([0, 0], [1, 0]);
      const c1 = k.compute([0, 0], [0, 1]);
      expect(c1).toBeGreaterThan(c0);
    });

    it("matches the isotropic kernel when the array is constant", () => {
      const ard = new Matern({ lengthScale: [1.5, 1.5], nu: 1.5 });
      const iso = new Matern({ lengthScale: 1.5, nu: 1.5 });
      expect(ard.compute([0, 0], [1, 2])).toBeCloseTo(iso.compute([0, 0], [1, 2]), 12);
    });
  });

  describe("DotProduct kernel", () => {
    it("computes sigma0^2 + x·x'", () => {
      const k = new DotProduct({ sigma0: 2 });
      expect(k.compute([1, 2], [3, 4])).toBeCloseTo(4 + (1 * 3 + 2 * 4), 12);
    });
  });

  describe("marginal likelihood optimization", () => {
    // y depends only on x0; x1 is pure noise -> ARD should make l[1] >> l[0].
    const r = rng(7);
    const X = [];
    const y = [];
    for (let i = 0; i < 40; i++) {
      const x0 = r() * 4 - 2;
      const x1 = r() * 4 - 2; // irrelevant
      X.push([x0, x1]);
      y.push(Math.sin(x0 * 1.5) + (r() - 0.5) * 0.05);
    }

    it("logMarginalLikelihood is finite and improves after optimization", () => {
      const fixed = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
        alpha: 1e-2,
      });
      fixed.fit(X, y);
      const llFixed = fixed.logMarginalLikelihood();
      expect(Number.isFinite(llFixed)).toBe(true);

      const tuned = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
        alpha: 1e-2,
        optimize: true,
        nRestarts: 1,
      });
      tuned.fit(X, y);
      // Optimization must not decrease the marginal likelihood.
      expect(tuned.logMarginalLikelihood()).toBeGreaterThanOrEqual(llFixed - 1e-6);
    });

    it("ARD down-weights the irrelevant feature (l[1] >> l[0])", () => {
      const gp = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
        alpha: 1e-2,
        optimize: true,
        nRestarts: 2,
      });
      gp.fit(X, y);
      const [l0, l1] = gp.kernel.lengthScale;
      expect(l1).toBeGreaterThan(l0 * 2);
    });

    it("improves predictive fit on held-out data after tuning", () => {
      const fixed = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
        alpha: 1e-2,
      }).fit(X, y);
      const tuned = new GaussianProcessRegressor({
        kernel: new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
        alpha: 1e-2,
        optimize: true,
        nRestarts: 2,
      }).fit(X, y);

      const xt = [[0.5, 0.0], [-1.0, 1.0], [1.2, -0.5]];
      const yt = xt.map((x) => Math.sin(x[0] * 1.5));
      const mse = (preds) =>
        preds.reduce((s, p, i) => s + (p - yt[i]) ** 2, 0) / preds.length;
      expect(mse(tuned.predict(xt))).toBeLessThanOrEqual(mse(fixed.predict(xt)) + 1e-9);
    });
  });

  describe("composite kernel optimization", () => {
    it("optimizes a Matérn(ARD) + DotProduct sum without error", () => {
      const r2 = rng(11);
      const X = [];
      const y = [];
      for (let i = 0; i < 30; i++) {
        const x0 = r2() * 4 - 2;
        const x1 = r2() * 4 - 2;
        X.push([x0, x1]);
        y.push(2 * x0 + Math.sin(x1)); // linear + nonlinear
      }
      const kernel = new SumKernel({
        kernels: [
          new Matern({ lengthScale: [1, 1], nu: 2.5, variance: 1 }),
          new DotProduct({ sigma0: 1 }),
        ],
      });
      const gp = new GaussianProcessRegressor({ kernel, alpha: 1e-2, optimize: true, nRestarts: 1 });
      gp.fit(X, y);
      expect(Number.isFinite(gp.logMarginalLikelihood())).toBe(true);
    });
  });

  describe("backward compatibility", () => {
    it("does not optimize by default (kernel params unchanged after fit)", () => {
      const k = new Matern({ lengthScale: 1.0, nu: 1.5, variance: 1.0 });
      const gp = new GaussianProcessRegressor({ kernel: k, alpha: 1e-2 });
      gp.fit([[0], [1], [2], [3]], [0, 1, 2, 3]);
      expect(gp.kernel.lengthScale).toBe(1.0);
    });
  });
});
