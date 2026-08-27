/**
 * Ridge (L2) regularization for GLM.
 *
 * The penalty must leave the INTERCEPT alone. It carries the scale of y rather
 * than the influence of a predictor, so shrinking it toward zero drags the
 * whole fit down and the slopes distort to compensate — on a target with a
 * large mean, badly enough to flip their signs. sklearn's Ridge, glmnet and
 * MASS::lm.ridge all exclude it.
 *
 * sklearn reference values below come from
 * `Ridge(alpha=..., fit_intercept=True, solver='cholesky')` on the fixture in
 * this file; regenerate by feeding X/Y below to that call.
 */

import { describe, expect, it } from 'vitest';
import { GLM } from '../src/stats/estimators/GLM.js';

// Predictors deliberately on different scales (~1, ~10, ~0.1): ridge here does
// not standardize, matching sklearn's default, and that has to stay pinned.
// The target has a large mean — the case the old implementation destroyed.
const X = [
  [1.0, 20.0, 0.10], [2.0, -5.0, 0.35], [3.0, 12.0, 0.05],
  [4.0, -18.0, 0.42], [5.0, 7.0, 0.18], [6.0, 25.0, 0.31],
  [7.0, -11.0, 0.07], [8.0, 3.0, 0.28], [9.0, -22.0, 0.39],
  [10.0, 15.0, 0.12], [11.0, -8.0, 0.45], [12.0, 9.0, 0.22],
];
const y = [
  203.1, 196.4, 202.5, 191.8, 200.7, 208.2,
  194.9, 200.3, 190.6, 206.1, 196.8, 203.4,
];

// [intercept, b1, b2, b3] per alpha, from sklearn.
const SKLEARN = {
  0: [196.7522132348555, 0.28996017237122357, 0.37099600398574634, 0.38763796910333415],
  1: [196.8403119287438, 0.28974635810273913, 0.36912934530645675, 0.05086687067570423],
  10: [196.97413832273557, 0.2716476789893988, 0.3668658614075889, 0.005592825848835456],
  100: [197.7109975981762, 0.16432165606092541, 0.3500509547960107, -0.00014834365936095489],
};

const coefs = (m) => m.model?.coefficients ?? m.coefficients;
const ridge = (alpha, params = {}) => {
  const m = new GLM({ family: 'gaussian', regularization: { alpha, l1_ratio: 0 }, ...params });
  m.fit(X, y);
  return coefs(m);
};

describe('GLM ridge regularization', () => {
  describe('matches sklearn Ridge', () => {
    it.each(Object.keys(SKLEARN))('alpha=%s', (alpha) => {
      ridge(Number(alpha)).forEach((c, i) => expect(c).toBeCloseTo(SKLEARN[alpha][i], 9));
    });

    it('alpha=0 reproduces the unpenalized fit', () => {
      const plain = new GLM({ family: 'gaussian' });
      plain.fit(X, y);
      ridge(0).forEach((c, i) => expect(c).toBeCloseTo(coefs(plain)[i], 10));
    });
  });

  describe('the intercept is not penalized', () => {
    it('leaves the intercept near the target mean however hard the penalty', () => {
      // Previously alpha=100 pulled a ~197 intercept down toward zero.
      const yMean = y.reduce((a, b) => a + b, 0) / y.length;
      for (const alpha of [1, 10, 100, 1000]) {
        expect(Math.abs(ridge(alpha)[0] - yMean)).toBeLessThan(5);
      }
    });

    it('does not blow the slopes up on a large-mean target', () => {
      // The regression this guards. With the intercept penalized, a target
      // with mean ~200 sent the slopes AWAY from their unpenalized values to
      // make up the lost offset — growing severalfold and changing sign. A
      // correct ridge only ever pulls them toward zero.
      //
      // (Ridge may still take an individually weak coefficient THROUGH zero as
      // it shrinks — sklearn's own reference does exactly that for b3 at
      // alpha=100 — so this asserts magnitudes, not signs.)
      const ols = ridge(0).slice(1);
      for (const alpha of [1, 10, 100]) {
        ridge(alpha).slice(1).forEach((c, i) => {
          expect(Math.abs(c)).toBeLessThanOrEqual(Math.abs(ols[i]) + 1e-9);
        });
      }
    });

    it('equals centring X and y, fitting without an intercept, and recovering it', () => {
      // The algebraic identity that an unpenalized intercept is meant to satisfy.
      const alpha = 7;
      const p = X[0].length;
      const xBar = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j], 0) / X.length);
      const yBar = y.reduce((a, b) => a + b, 0) / y.length;
      const centered = new GLM({
        family: 'gaussian',
        intercept: false,
        regularization: { alpha, l1_ratio: 0 },
      });
      centered.fit(X.map((r) => r.map((v, j) => v - xBar[j])), y.map((v) => v - yBar));

      const slopes = coefs(centered);
      const recovered = yBar - slopes.reduce((s, b, j) => s + b * xBar[j], 0);
      const direct = ridge(alpha);
      expect(direct[0]).toBeCloseTo(recovered, 8);
      slopes.forEach((b, j) => expect(direct[j + 1]).toBeCloseTo(b, 8));
    });

    it('penalizes every column when there is no intercept', () => {
      // sklearn: Ridge(alpha=5, fit_intercept=False)
      const expected = [22.898519284138867, 1.8077550284843515, 21.178500326506576];
      ridge(5, { intercept: false }).forEach((c, i) => expect(c).toBeCloseTo(expected[i], 9));
    });
  });

  describe('shrinkage behaviour', () => {
    it('shrinks the slopes monotonically as alpha grows', () => {
      const norm = (a) => Math.hypot(...a.slice(1));
      const norms = [0, 1, 10, 100, 1000].map((a) => norm(ridge(a)));
      for (let i = 1; i < norms.length; i++) expect(norms[i]).toBeLessThan(norms[i - 1]);
    });
  });

  describe('option handling', () => {
    it('ignores a top-level alpha — that is the CI significance level', () => {
      // `alpha` at the top level sets confidence-interval coverage, NOT a
      // penalty. Regularization has to be nested under `regularization`.
      const top = new GLM({ family: 'gaussian', alpha: 100, l1_ratio: 0 });
      top.fit(X, y);
      const plain = new GLM({ family: 'gaussian' });
      plain.fit(X, y);
      coefs(top).forEach((c, i) => expect(c).toBeCloseTo(coefs(plain)[i], 12));
    });

    it('rejects an L1 component with a message pointing at ridge', () => {
      expect(() => ridge(1, { regularization: { alpha: 1, l1_ratio: 0.5 } }))
        .toThrow(/L1 regularization is not implemented/);
    });
  });
});
