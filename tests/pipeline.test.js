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

  it('should spread fold remainders evenly and reuse one split across candidates', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7]];
    const y = [1, 2, 3, 4, 5, 6, 7];
    const seenByOffset = new Map();

    const grid = new GridSearchCV(
      (params) => ({
        offset: params.offset,
        fit(_X, yTrain) {
          if (yTrain.length < y.length) {
            if (!seenByOffset.has(params.offset)) {
              seenByOffset.set(params.offset, []);
            }
            seenByOffset.get(params.offset).push(yTrain);
          }
          return this;
        },
        predict(XTest) {
          return XTest.map(([v]) => v + this.offset);
        },
      }),
      { offset: [0, 1] },
      (yTrue, yPred) => -yTrue.reduce((acc, v, i) => acc + Math.abs(v - yPred[i]), 0),
      3,
    );

    grid.fit(X, y);

    const baseline = seenByOffset.get(0);
    // 7 samples across 3 folds => test sizes [3, 2, 2], train sizes [4, 5, 5]
    expect(baseline.map((arr) => arr.length)).toEqual([4, 5, 5]);
    expect(seenByOffset.get(1)).toEqual(baseline);
  });
});
