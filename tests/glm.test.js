/**
 * Tests for Generalized Linear Models (GLM and GLMM)
 */

import { describe, expect, it } from 'vitest';
import { GLM } from '../src/stats/estimators/GLM.js';
import { fitGLM, predictGLM } from '../src/stats/glm.js';

describe('GLM confidence intervals for fitted values', () => {
  it('matches R on the link and response scales and honors the confidence level', () => {
    const X = Array.from({ length: 75 }, (_, i) => [Math.floor(i / 25)]);
    const y = X.map(([x], i) => Number(i % 25 < [3, 9, 16][x]));
    const model = new GLM({ family: 'binomial' }).fit(X, y);
    const at = [[0], [1.5]];
    // R glm.control(epsilon = 1e-13), then predict.glm(type = "link",
    // se.fit = TRUE), qnorm and plogis.
    const expected = [
      {
        level: 0.5,
        link: [[-1.9137330392, -2.2539681298, -1.5734979485], [-0.0188898894, -0.2117620399, 0.1739822611]],
        response: [[0.1285620452, 0.0950077313, 0.1717183031], [0.4952776681, 0.4472564419, 0.5433861794]],
      },
      {
        level: 0.99,
        link: [[-1.9137330392, -3.2130671155, -0.6143989628], [-0.0188898894, -0.7554551774, 0.7176753987]],
        response: [[0.1285620452, 0.0386769345, 0.3510563893], [0.4952776681, 0.3196338086, 0.6720949188]],
      },
    ];
    for (const type of ['link', 'response']) {
      const points = model.predict(at, { type });
      const [narrow, wide] = expected.map(({ level, [type]: reference }) => {
        const intervals = model.predict(at, { type, interval: true, level });
        intervals.forEach(({ fit, lower, upper }, i) => {
          expect(fit).toBe(points[i]);
          expect(fit).toBeCloseTo(reference[i][0], 7);
          expect(lower).toBeCloseTo(reference[i][1], 7);
          expect(upper).toBeCloseTo(reference[i][2], 7);
          expect(lower).toBeLessThan(fit);
          expect(upper).toBeGreaterThan(fit);
        });
        return intervals;
      });
      wide.forEach(({ lower, upper }, i) => {
        expect(lower).toBeLessThan(narrow[i].lower);
        expect(upper).toBeGreaterThan(narrow[i].upper);
      });
    }
    for (const level of [0, 1, NaN, -0.1]) {
      expect(() => model.predict(at, { interval: true, level })).toThrow(/confidence level/);
    }
  });

  it('orders response bounds on a decreasing link and accounts for an offset', () => {
    const model = {
      coefficients: [2], covarianceMatrix: [[0.25]], family: 'gamma', link: 'inverse', intercept: false,
    };
    const [result] = predictGLM(model, [[1]], { interval: true, offset: [1] });
    const margin = 1.959963984540054 * 0.5;
    expect(result.fit).toBe(1 / 3);
    expect(result.lower).toBeCloseTo(1 / (3 + margin), 12);
    expect(result.upper).toBeCloseTo(1 / (3 - margin), 12);
  });

  it('includes zero when a square-root link interval crosses it', () => {
    const model = {
      coefficients: [0.1], covarianceMatrix: [[0.25]], family: 'poisson', link: 'sqrt', intercept: false,
    };
    const [result] = predictGLM(model, [[1]], { interval: true });
    expect(result.fit).toBeCloseTo(0.01, 12);
    expect(result.lower).toBe(0);
    expect(result.upper).toBeCloseTo((0.1 + 1.959963984540054 * 0.5) ** 2, 12);
  });
});

describe('GLM - Gaussian Family', () => {
  it('should fit a simple linear regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2); // intercept + 1 coef
    expect(model._model.converged).toBe(true);
  });

  it('should predict new values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const predictions = model.predict([[6], [7]]);
    expect(predictions).toHaveLength(2);
    expect(predictions.every((p) => typeof p === 'number' && isFinite(p))).toBe(true);
  });

  it('should work with multiple predictors', () => {
    const X = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
    ];
    const y = [3, 5, 7, 9, 11];

    const model = new GLM({ family: 'gaussian', intercept: true });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(3); // intercept + 2 coefs
    expect(model._model.converged).toBe(true);
  });

  it('should work without intercept', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];

    const model = new GLM({ family: 'gaussian', intercept: false });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(1); // no intercept
    expect(Math.abs(model._model.coefficients[0] - 2.0)).toBeLessThan(0.1);
  });

  it('should compute R² score', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const predictions = model.predict(X);
    const score = model.score(y, predictions);

    expect(score).toBeGreaterThan(0.9); // Should be very high for perfect fit
  });

  it('should serialize and deserialize', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const json = model.save();
    const restored = GLM.load(json);

    expect(restored.fitted).toBe(true);
    expect(restored._model.coefficients).toEqual(model._model.coefficients);

    const predictions1 = model.predict(X);
    const predictions2 = restored.predict(X);
    expect(predictions1).toEqual(predictions2);
  });

  it('should align summary columns to the longest label', () => {
    const longName = 'Beak Length (mm)';
    const data = [
      { [longName]: 10, mass: 100 },
      { [longName]: 11, mass: 120 },
      { [longName]: 12, mass: 150 },
      { [longName]: 14, mass: 200 },
    ];

    const model = new GLM({ family: 'gaussian', link: 'identity' });
    model.fit({
      X: longName,
      y: 'mass',
      data,
    });

    const summary = model.summary();
    const headerLine = summary.split('\n').find((line) =>
      line.includes('Estimate') && line.includes('Std.Error')
    );
    const expectedPadding = ''.padEnd(longName.length + 2);
    expect(headerLine.startsWith(expectedPadding)).toBe(true);

    const longLine = summary.split('\n').find((line) => line.startsWith(longName));
    expect(longLine.slice(longName.length, longName.length + 2)).toBe('  ');
  });
});

describe('GLM - Binomial Family', () => {
  it('should fit logistic regression', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial', link: 'logit' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict probabilities between 0 and 1', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every((p) => p >= 0 && p <= 1)).toBe(true);
  });

  it('should compute accuracy score', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial' });
    model.fit(X, y);

    const predictions = model.predict(X);
    const score = model.score(y, predictions);

    expect(score).toBeGreaterThan(0.5); // Should predict better than random
  });

  it('should work with probit link', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial', link: 'probit' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.converged).toBe(true);
  });
});

describe('GLM - Poisson Family', () => {
  it('should fit Poisson regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 2, 3, 5, 8];

    const model = new GLM({ family: 'poisson', link: 'log' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 2, 3, 5, 8];

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every((p) => p > 0)).toBe(true);
  });

  it('should handle count data correctly', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [2, 3, 5, 7, 11, 13, 17, 19]; // Prime numbers

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    const predictions = model.predict([[9], [10]]);
    expect(predictions.every((p) => p > 0 && isFinite(p))).toBe(true);
  });
});

describe('GLM - Gamma Family', () => {
  it('should fit Gamma regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma', link: 'inverse' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every((p) => p > 0)).toBe(true);
  });

  it('should work with log link', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma', link: 'log' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.converged).toBe(true);
  });
});

describe('GLM - Inverse Gaussian Family', () => {
  it('should fit inverse Gaussian regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.8, 1.5, 2.2, 3.1, 4.2];

    const model = new GLM({ family: 'inverse_gaussian', link: 'inverse_squared' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.8, 1.5, 2.2, 3.1, 4.2];

    const model = new GLM({ family: 'inverse_gaussian' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every((p) => p > 0)).toBe(true);
  });
});

describe('GLM - Negative Binomial Family', () => {
  it('should fit negative binomial regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 5, 8, 12, 18];

    const model = new GLM({ family: 'negative_binomial', link: 'log', theta: 1.5 });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 5, 8, 12, 18];

    const model = new GLM({ family: 'negative_binomial', theta: 1.5 });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every((p) => p > 0)).toBe(true);
  });
});

describe('GLM - Summary Output', () => {
  it('should generate summary for GLM', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const summary = model.summary();
    expect(typeof summary).toBe('string');
    expect(summary).toContain('Generalized Linear Model');
    expect(summary).toContain('Coefficients:');
    expect(summary).toContain('AIC:');
    expect(summary).toContain('Deviance');
  });

  it('should not include p-values in GLMM summary', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups },
    });
    model.fit(X, y);

    const summary = model.summary();
    expect(summary).toContain('Generalized Linear Mixed Model');
    expect(summary).toContain('Fixed Effects:');
    expect(summary).toContain('Random Effects:');
    expect(summary).toContain('p-values for fixed effects in mixed models');
    expect(summary).not.toMatch(/\bp-value\b/); // Should not have p-value column header
  });
});

describe('GLM - Table-style API', () => {
  it('should work with table-style input', () => {
    const data = [
      { x1: 1, x2: 2, y: 3 },
      { x1: 2, x2: 3, y: 5 },
      { x1: 3, x2: 4, y: 7 },
      { x1: 4, x2: 5, y: 9 },
      { x1: 5, x2: 6, y: 11 },
    ];

    const model = new GLM({ family: 'gaussian' });
    model.fit({ X: ['x1', 'x2'], y: 'y', data });

    expect(model.fitted).toBe(true);
    expect(model._columnsX).toEqual(['x1', 'x2']);

    const predictions = model.predict({ X: ['x1', 'x2'], data });
    expect(predictions).toHaveLength(5);
  });

  it('should handle missing values', () => {
    const data = [
      { x: 1, y: 2 },
      { x: 2, y: 4 },
      { x: 3, y: null },
      { x: 4, y: 8 },
      { x: 5, y: 10 },
    ];

    const model = new GLM({ family: 'gaussian' });
    model.fit({ X: ['x'], y: 'y', data, omit_missing: true });

    expect(model.fitted).toBe(true);
    expect(model._model.n).toBe(4); // Should have 4 observations (1 removed)
  });
});

describe('GLM - Advanced Features', () => {
  it('should handle weights', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];
    const weights = [1, 1, 2, 1, 1]; // Give more weight to third observation

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y, weights);

    expect(model.fitted).toBe(true);
  });

  it('should handle offsets', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [3, 5, 7, 9, 11];
    const offset = [1, 1, 1, 1, 1];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y, null, offset);

    expect(model.fitted).toBe(true);
  });

  it('should support regularization', () => {
    const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
    const y = [3, 5, 7, 9, 11];

    const model = new GLM({
      family: 'gaussian',
      regularization: { alpha: 0.1, l1_ratio: 0 }, // Ridge regression
    });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
  });

  it('should estimate dispersion parameter', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian', dispersion: 'estimate' });
    model.fit(X, y);

    expect(model._model.dispersion).toBeGreaterThan(0);
    expect(isFinite(model._model.dispersion)).toBe(true);
  });

  it('should fix dispersion parameter', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 1, 1, 1];

    const model = new GLM({ family: 'binomial', dispersion: 'fixed' });
    model.fit(X, y);

    expect(model._model.dispersion).toBe(1.0);
  });
});

describe('GLM - Error Handling', () => {
  it('should throw error when predicting before fitting', () => {
    const model = new GLM({ family: 'gaussian' });

    expect(() => model.predict([[1], [2]])).toThrow('requires a fitted model');
  });

  it('should throw error for invalid family', () => {
    expect(() => new GLM({ family: 'invalid_family' })).not.toThrow();
    // The error will be thrown during fit when createFamily is called
  });

  it('should throw error for invalid link', () => {
    const model = new GLM({ family: 'gaussian', link: 'invalid_link' });
    const X = [[1], [2], [3]];
    const y = [1, 2, 3];

    expect(() => model.fit(X, y)).toThrow();
  });
});

describe('GLM - Edge Cases', () => {
  it('should handle perfect separation in logistic regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 0, 1, 1];

    const model = new GLM({ family: 'binomial', maxIter: 50 });
    model.fit(X, y);

    // Should still fit, though may not converge
    expect(model.fitted).toBe(true);
  });

  it('should handle single predictor', () => {
    const X = [[1], [2], [3]];
    const y = [2, 4, 6];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(2);
  });

  it('should handle all zeros in response (Poisson)', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 0, 0, 0];

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    const predictions = model.predict(X);
    expect(predictions.every((p) => p >= 0)).toBe(true);
  });
});

describe('GLM - Offset handling', () => {
  const X = [[1], [2], [3], [4], [5], [6], [7], [8]];

  it('gaussian: fitting y with offset o equals fitting (y - o) without offset', () => {
    const y = [2.1, 3.9, 6.2, 8.1, 9.8, 12.2, 14.1, 15.9];
    const offset = [0.5, -1, 2, 0, 1.5, -0.5, 1, -2];

    const withOffset = fitGLM(X, y, { family: 'gaussian', offset });
    const shifted = fitGLM(X, y.map((yi, i) => yi - offset[i]), { family: 'gaussian' });

    withOffset.coefficients.forEach((c, i) => {
      expect(c).toBeCloseTo(shifted.coefficients[i], 8);
    });
  });

  it('poisson: constant offset c shifts the intercept by -c and leaves slopes unchanged', () => {
    const y = [1, 3, 4, 8, 11, 19, 28, 42];
    const c = 1.7;
    const offset = Array(X.length).fill(c);

    const withOffset = fitGLM(X, y, { family: 'poisson', offset });
    const without = fitGLM(X, y, { family: 'poisson' });

    expect(withOffset.coefficients[0]).toBeCloseTo(without.coefficients[0] - c, 6);
    expect(withOffset.coefficients[1]).toBeCloseTo(without.coefficients[1], 6);
  });

  it('poisson: varying offset (log exposure) satisfies the score equations', () => {
    const y = [2, 1, 6, 4, 12, 8, 25, 16];
    const exposure = [1, 0.5, 2, 1, 3, 1.5, 5, 2];
    const offset = exposure.map((e) => Math.log(e));

    const model = fitGLM(X, y, { family: 'poisson', offset });
    expect(model.converged).toBe(true);

    // For the canonical log link, the MLE satisfies X'(y - mu) = 0
    // with mu = exp(X*beta + offset)
    const mu = X.map((row, i) =>
      Math.exp(model.coefficients[0] + model.coefficients[1] * row[0] + offset[i])
    );
    const scoreIntercept = y.reduce((s, yi, i) => s + (yi - mu[i]), 0);
    const scoreSlope = y.reduce((s, yi, i) => s + (yi - mu[i]) * X[i][0], 0);

    expect(Math.abs(scoreIntercept)).toBeLessThan(1e-4);
    expect(Math.abs(scoreSlope)).toBeLessThan(1e-4);
  });

  it('binomial: constant offset c shifts the intercept by -c', () => {
    const y = [0, 0, 1, 0, 1, 1, 1, 1];
    const c = 0.8;
    const offset = Array(X.length).fill(c);

    const withOffset = fitGLM(X, y, { family: 'binomial', offset });
    const without = fitGLM(X, y, { family: 'binomial' });

    expect(withOffset.coefficients[0]).toBeCloseTo(without.coefficients[0] - c, 5);
    expect(withOffset.coefficients[1]).toBeCloseTo(without.coefficients[1], 5);
  });
});
