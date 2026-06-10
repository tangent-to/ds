/**
 * Tests for Gradient Boosting estimators
 */

import { describe, expect, it } from 'vitest';
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from '../src/ml/estimators/GradientBoosting.js';

function makeQuadratic(n = 80) {
  const X = Array.from({ length: n }, (_, i) => [(i - n / 2) / 10]);
  const y = X.map(([x]) => x * x);
  return { X, y };
}

describe('GradientBoostingRegressor', () => {
  it('learns a nonlinear function', () => {
    const { X, y } = makeQuadratic();
    const gbr = new GradientBoostingRegressor({
      nEstimators: 150,
      learningRate: 0.1,
      maxDepth: 3,
      seed: 42,
    });
    gbr.fit(X, y);

    const pred = gbr.predict([[2], [-3], [0]]);
    expect(pred[0]).toBeCloseTo(4, 0);
    expect(pred[1]).toBeCloseTo(9, 0);
    expect(Math.abs(pred[2])).toBeLessThan(0.5);
  });

  it('training loss decreases monotonically overall', () => {
    const { X, y } = makeQuadratic();
    const gbr = new GradientBoostingRegressor({ nEstimators: 100, seed: 1 });
    gbr.fit(X, y);

    const lh = gbr.lossHistory;
    expect(lh).toHaveLength(100);
    expect(lh[10]).toBeLessThan(lh[0]);
    expect(lh[99]).toBeLessThan(lh[10]);
  });

  it('more estimators fit better than one', () => {
    const { X, y } = makeQuadratic();
    const one = new GradientBoostingRegressor({ nEstimators: 1, seed: 1 });
    const many = new GradientBoostingRegressor({ nEstimators: 100, seed: 1 });
    one.fit(X, y);
    many.fit(X, y);

    const mse = (model) =>
      model.predict(X).reduce((s, p, i) => s + (p - y[i]) ** 2, 0) / y.length;
    expect(mse(many)).toBeLessThan(mse(one));
  });

  it('is reproducible with a seed when subsampling', () => {
    const { X, y } = makeQuadratic();
    const a = new GradientBoostingRegressor({ nEstimators: 30, subsample: 0.6, seed: 7 });
    const b = new GradientBoostingRegressor({ nEstimators: 30, subsample: 0.6, seed: 7 });
    a.fit(X, y);
    b.fit(X, y);

    expect(a.predict([[1.5], [-2.5]])).toEqual(b.predict([[1.5], [-2.5]]));
  });

  it('exposes normalized feature importances favouring informative features', () => {
    // y depends only on the first feature; second is constant-ish noise
    const X = Array.from({ length: 60 }, (_, i) => [i / 10, (i % 3) * 0.01]);
    const y = X.map(([x]) => 2 * x + 1);
    const gbr = new GradientBoostingRegressor({ nEstimators: 30, seed: 3 });
    gbr.fit(X, y);

    const imp = gbr.featureImportances;
    expect(imp).toHaveLength(2);
    expect(imp.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(imp[0]).toBeGreaterThan(imp[1]);
  });

  it('supports declarative {X, y, data} fitting and prediction', () => {
    const data = Array.from({ length: 50 }, (_, i) => ({
      size: i / 5,
      mass: 3 * (i / 5) + 2,
    }));

    const gbr = new GradientBoostingRegressor({ nEstimators: 80, seed: 5 });
    gbr.fit({ X: ['size'], y: 'mass', data });

    const pred = gbr.predict({ X: ['size'], data: [{ size: 5 }] });
    expect(pred[0]).toBeCloseTo(17, 0);
  });
});

describe('GradientBoostingClassifier', () => {
  function makeBinary(n = 60) {
    const X = [];
    const y = [];
    for (let i = 0; i < n; i++) {
      const x = i / 10 - n / 20;
      X.push([x, x * 0.5 + (i % 7) * 0.1]);
      y.push(x + (i % 7) * 0.05 > 0 ? 'pos' : 'neg');
    }
    return { X, y };
  }

  function makeThreeClass(n = 90) {
    const X = [];
    const y = [];
    const labels = ['a', 'b', 'c'];
    for (let i = 0; i < n; i++) {
      const k = i % 3;
      X.push([k * 2 + (i % 5) * 0.2, -k + (i % 4) * 0.1]);
      y.push(labels[k]);
    }
    return { X, y };
  }

  it('separates binary classes with string labels', () => {
    const { X, y } = makeBinary();
    const gbc = new GradientBoostingClassifier({ nEstimators: 50, seed: 1 });
    gbc.fit(X, y);

    expect(gbc.classes_).toEqual(['neg', 'pos']);
    const acc = gbc.predict(X).filter((p, i) => p === y[i]).length / y.length;
    expect(acc).toBeGreaterThan(0.95);
  });

  it('returns valid binary probabilities keyed by class label', () => {
    const { X, y } = makeBinary();
    const gbc = new GradientBoostingClassifier({ nEstimators: 30, seed: 1 });
    gbc.fit(X, y);

    const proba = gbc.predictProba([[2, 1], [-2, -1]]);
    for (const dist of proba) {
      expect(dist.pos + dist.neg).toBeCloseTo(1, 9);
      expect(dist.pos).toBeGreaterThanOrEqual(0);
      expect(dist.pos).toBeLessThanOrEqual(1);
    }
    expect(proba[0].pos).toBeGreaterThan(0.5);
    expect(proba[1].neg).toBeGreaterThan(0.5);
  });

  it('handles three classes with multinomial deviance', () => {
    const { X, y } = makeThreeClass();
    const gbc = new GradientBoostingClassifier({ nEstimators: 40, seed: 7 });
    gbc.fit(X, y);

    const acc = gbc.predict(X).filter((p, i) => p === y[i]).length / y.length;
    expect(acc).toBeGreaterThan(0.95);

    const probs = gbc.predictProba([[4, -2]])[0];
    const total = Object.values(probs).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 9);
  });

  it('handles numeric non-index labels', () => {
    const X = [[0], [0.1], [0.2], [5], [5.1], [5.2]];
    const y = [1, 1, 1, 2, 2, 2];

    const gbc = new GradientBoostingClassifier({ nEstimators: 20, seed: 1 });
    gbc.fit(X, y);

    expect(gbc.classes_).toEqual([1, 2]);
    expect(gbc.predict([[0.05], [5.05]])).toEqual([1, 2]);
  });

  it('training deviance decreases', () => {
    const { X, y } = makeBinary();
    const gbc = new GradientBoostingClassifier({ nEstimators: 40, seed: 1 });
    gbc.fit(X, y);

    const lh = gbc.lossHistory;
    expect(lh[39]).toBeLessThan(lh[0]);
  });

  it('supports declarative {X, y, data} fitting with label decoding', () => {
    const data = Array.from({ length: 60 }, (_, i) => ({
      bill: i / 10,
      species: i < 30 ? 'adelie' : 'gentoo',
    }));

    const gbc = new GradientBoostingClassifier({ nEstimators: 30, seed: 2 });
    gbc.fit({ X: ['bill'], y: 'species', data });

    const pred = gbc.predict({ X: ['bill'], data: [{ bill: 0.5 }, { bill: 5.5 }] });
    expect(pred).toEqual(['adelie', 'gentoo']);
  });
});
