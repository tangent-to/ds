import { describe, expect, it } from 'vitest';
import { GridSearchCV } from '../src/ml/pipeline.js';

describe('GridSearchCV (pipeline)', () => {
  it('should accept table descriptors for fit()', () => {
    const data = [
      { feature: 1, target: 1 },
      { feature: 2, target: 2 },
      { feature: 3, target: 3 },
      { feature: 4, target: 4 },
    ];

    const grid = new GridSearchCV(
      (params) => ({
        offset: params.offset,
        mean: 0,
        fit(X, y) {
          this.mean = y.reduce((sum, val) => sum + val, 0) / y.length;
          return this;
        },
        predict(X) {
          return X.map(() => this.mean + this.offset);
        },
      }),
      { offset: [0, 0.5] },
      (yTrue, yPred) => -yTrue.reduce((acc, val, idx) => acc + Math.abs(val - yPred[idx]), 0),
      2,
    );

    grid.fit({ data, X: ['feature'], y: 'target' });

    expect(grid.bestParams).toHaveProperty('offset');
    expect(grid.bestEstimator).not.toBeNull();
    const preds = grid.predict([[10]]);
    expect(Array.isArray(preds)).toBe(true);
    expect(preds).toHaveLength(1);
  });
});
