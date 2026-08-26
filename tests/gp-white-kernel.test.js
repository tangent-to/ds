/**
 * WhiteKernel: the learnable counterpart to the regressor's fixed `alpha`.
 *
 * `alpha` is noise you already know and state (a poll's sampling variance);
 * a WhiteKernel is noise estimated from the data by `optimize: true`. This is
 * the same split scikit-learn draws, and these tests pin both halves of it.
 *
 * sklearn reference numbers come from `src/ml/test-gp-vs-sklearn.py`
 * ("Test 9: WhiteKernel vs alpha"); re-run that script to regenerate them.
 */

import { describe, expect, it } from "vitest";
import { GaussianProcessRegressor } from "../src/ml/estimators/GaussianProcessRegressor.js";
import { Matern, RBF, SumKernel, WhiteKernel } from "../src/ml/kernels/index.js";

const X = [[0], [1], [2], [3], [4], [5]];
const y = [0.0, 0.8, 0.2, 1.4, 0.1, 1.1];
const X_TEST = [[0], [0.5], [1], [2], [2.5], [4], [5]];
const NOISE = 0.1;

// sklearn: RBF(1.5) + WhiteKernel(0.1), alpha=0, optimizer=None
const SKLEARN_WHITE_MEAN = [
  0.13172329398354712, 0.3031579942348093, 0.45834419196519116,
  0.6952942205868653, 0.7506414838133253, 0.6454786855633365,
  0.7971460853642416,
];
const SKLEARN_WHITE_STD = [
  0.42207651484818354, 0.39598474925807464, 0.39683432155507614,
  0.3943024431133655, 0.3941034599517599, 0.3968343215550763,
  0.4220765148481834,
];
const SKLEARN_WHITE_LML = -10.559550530375517;

const sumKernel = (...kernels) => new SumKernel({ kernels });
const whiteGP = () =>
  new GaussianProcessRegressor({
    kernel: sumKernel(new RBF(1.5, 1.0), new WhiteKernel(NOISE)),
    alpha: 0,
  });

describe("WhiteKernel", () => {
  describe("covariance structure", () => {
    it("is noiseLevel·I on K(X, X), even for duplicated inputs", () => {
      // Two readings at the same input are still independent measurements —
      // this is what makes white noise not a function of the input values.
      const K = new WhiteKernel(0.3).call([[0], [1], [0]]).to2DArray();
      expect(K).toEqual([
        [0.3, 0, 0],
        [0, 0.3, 0],
        [0, 0, 0.3],
      ]);
    });

    it("is all zeros on a cross-covariance, including coinciding points", () => {
      // A test point at x=0 must not inherit the training point's noise.
      const K = new WhiteKernel(0.3).call([[0], [1]], [[0], [5]]).to2DArray();
      expect(K).toEqual([
        [0, 0],
        [0, 0],
      ]);
    });

    it("reports noiseLevel as the variance at a point", () => {
      const w = new WhiteKernel(0.3);
      const x = [0];
      expect(w.compute(x, x)).toBe(0.3);
    });

    it("round-trips through getParams/setParams/clone", () => {
      const w = new WhiteKernel({ noise_level: 0.25 });
      expect(w.getParams()).toEqual({ noiseLevel: 0.25 });
      w.setParams({ noiseLevel: 0.5 });
      expect(w.noiseLevel).toBe(0.5);
      expect(w.clone().noiseLevel).toBe(0.5);
    });
  });

  describe("composition inside a SumKernel", () => {
    // SumKernel.call() must sum child MATRICES; summing pointwise compute()
    // values would drop the noise term or leak it into the train/test block.
    const kernel = sumKernel(new RBF(1.5, 1.0), new WhiteKernel(NOISE));

    it("adds the noise to the diagonal of K(X, X) only", () => {
      const K = kernel.call(X).to2DArray();
      const rbf = new RBF(1.5, 1.0).call(X).to2DArray();
      for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X.length; j++) {
          expect(K[i][j]).toBeCloseTo(rbf[i][j] + (i === j ? NOISE : 0), 12);
        }
      }
    });

    it("leaves the cross-covariance free of noise", () => {
      const K = kernel.call(X, X_TEST).to2DArray();
      const rbf = new RBF(1.5, 1.0).call(X, X_TEST).to2DArray();
      K.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(rbf[i][j], 12)));
    });
  });

  describe("matches sklearn", () => {
    it("reproduces the posterior mean and std of RBF + WhiteKernel", () => {
      const { mean, std } = whiteGP().fit(X, y).predict(X_TEST, { returnStd: true });
      mean.forEach((m, i) => expect(m).toBeCloseTo(SKLEARN_WHITE_MEAN[i], 10));
      std.forEach((s, i) => expect(s).toBeCloseTo(SKLEARN_WHITE_STD[i], 10));
    });

    it("reproduces the log marginal likelihood", () => {
      expect(whiteGP().fit(X, y).logMarginalLikelihood_).toBeCloseTo(SKLEARN_WHITE_LML, 10);
    });
  });

  describe("relationship to alpha", () => {
    // Both put the same numbers on the diagonal of K, so they give the same
    // posterior mean and the same likelihood. They differ only in what the
    // predictive std describes — and in whether the optimizer may move them.
    const viaWhite = () => whiteGP().fit(X, y);
    const viaAlpha = () =>
      new GaussianProcessRegressor({ kernel: new RBF(1.5, 1.0), alpha: NOISE }).fit(X, y);

    it("gives the same posterior mean and likelihood as an equal alpha", () => {
      const w = viaWhite();
      const a = viaAlpha();
      w.predict(X_TEST).forEach((m, i) => expect(m).toBeCloseTo(a.predict(X_TEST)[i], 12));
      expect(w.logMarginalLikelihood_).toBeCloseTo(a.logMarginalLikelihood_, 12);
    });

    it("predicts a new observation, where alpha predicts the latent function", () => {
      // var_white = var_alpha + noiseLevel, exactly. sklearn behaves the same.
      const { std: sw } = viaWhite().predict(X_TEST, { returnStd: true });
      const { std: sa } = viaAlpha().predict(X_TEST, { returnStd: true });
      sw.forEach((s, i) => expect(s * s - sa[i] * sa[i]).toBeCloseTo(NOISE, 10));
    });
  });

  describe("optimization", () => {
    it.each([
      ["derivative-free (RBF)", () => new RBF(1.0, 1.0)],
      ["analytic gradients (Matérn)", () => new Matern({ lengthScale: 1.0, nu: 2.5, amplitude: 1.0 })],
    ])("%s: learns noiseLevel while alpha stays put", (_name, signal) => {
      const white = new WhiteKernel(0.5);
      const gp = new GaussianProcessRegressor({
        kernel: sumKernel(signal(), white),
        alpha: 1e-8,
        optimize: true,
      });
      gp.fit(X, y);

      expect(white.noiseLevel).not.toBe(0.5);
      expect(white.noiseLevel).toBeGreaterThan(0);
      expect(gp.alpha).toBe(1e-8);
      // The reported likelihood is the one that was maximized.
      expect(gp.logMarginalLikelihood()).toBeCloseTo(gp.logMarginalLikelihood_, 10);
    });

    it("beats its own neighbours in noiseLevel", () => {
      const white = new WhiteKernel(0.5);
      const gp = new GaussianProcessRegressor({
        kernel: sumKernel(new RBF(1.0, 1.0), white),
        alpha: 1e-8,
        optimize: true,
      });
      gp.fit(X, y);
      const best = gp.logMarginalLikelihood_;
      const tuned = { ls: gp.kernel.kernels[0].lengthScale, v: gp.kernel.kernels[0].variance, n: white.noiseLevel };

      for (const factor of [0.5, 2.0]) {
        const probe = new GaussianProcessRegressor({
          kernel: sumKernel(new RBF(tuned.ls, tuned.v), new WhiteKernel(tuned.n * factor)),
          alpha: 1e-8,
        });
        probe.fit(X, y);
        expect(probe.logMarginalLikelihood_).toBeLessThan(best + 1e-6);
      }
    });

    it("keeps the Matérn + WhiteKernel sum on the fast analytic path", () => {
      const gp = new GaussianProcessRegressor({
        kernel: sumKernel(new Matern({ lengthScale: 1.0, nu: 2.5 }), new WhiteKernel(0.1)),
      });
      gp.fit(X, y);
      expect(gp._analyticParts()).not.toBeNull();
      // An unsupported term drops the whole kernel back to the pattern search.
      const rq = new GaussianProcessRegressor({
        kernel: sumKernel(new RBF(1.0, 1.0), new WhiteKernel(0.1)),
      });
      rq.fit(X, y);
      expect(rq._analyticParts()).toBeNull();
    });
  });
});
