import { describe, it, expect } from 'vitest';
import { MLPRegressor } from '../src/ml/index.js';
import { Pipeline } from '../src/ml/pipeline.js';
import { StandardScaler } from '../src/ml/preprocessing.js';

const X = Array.from({ length: 40 }, (_, i) => [i / 10, (i % 7) / 7]);
const y = X.map(([a, b]) => 2 * a - b + 0.5);

describe('MLPRegressor on nn', () => {
  it('fits a linear map with L-BFGS and predicts a flat array', () => {
    const m = new MLPRegressor({ layerSizes: [2, 1], optimizer: 'lbfgs', epochs: 100, seed: 1 });
    m.fit(X, y);
    const pred = m.predict([[1, 0.5]]);
    expect(pred).toHaveLength(1);
    expect(pred[0]).toBeCloseTo(2 - 0.5 + 0.5, 3);
    expect(m.evaluate(X, y)).toBeLessThan(1e-4);
    expect(m.predictGradient([1, 0.5]).map((g) => +g.toFixed(3))).toEqual([2, -1]);
  });

  it('defaults to one hidden layer, trains with Adam, and reports a summary', () => {
    const m = new MLPRegressor({ epochs: 30, learningRate: 0.02, seed: 2 });
    m.fit(X, y);
    const s = m.summary();
    expect(s.layerSizes).toEqual([2, 4, 1]);
    expect(s.epochs).toBe(30);
    expect(s.losses).toHaveLength(30);
    expect(s.finalLoss).toBeLessThan(s.initialLoss);
    expect(s.network).toContain('dense_1');
  });

  it('checks layerSizes against the data', () => {
    expect(() => new MLPRegressor({ layerSizes: [3, 4, 1] }).fit(X, y)).toThrow(/X has 2 features/);
    expect(() => new MLPRegressor({ layerSizes: [2, 4, 2] }).fit(X, y)).toThrow(/y has 1 column/);
  });

  it('dropout gives Monte Carlo intervals, and a seed makes them repeatable', () => {
    const m = new MLPRegressor({ layerSizes: [2, 8, 1], dropout: 0.2, epochs: 20, seed: 3 });
    m.fit(X, y);
    const a = m.predict(X.slice(0, 3), { samples: 20, seed: 1 });
    const b = m.predict(X.slice(0, 3), { samples: 20, seed: 1 });
    expect(a.mean).toEqual(b.mean);
    expect(Math.max(...a.epistemic)).toBeGreaterThan(0);
  });

  it('sits in a Pipeline and round-trips through JSON', () => {
    const p = new Pipeline([new StandardScaler(), new MLPRegressor({ layerSizes: [2, 1], optimizer: 'lbfgs', epochs: 50, seed: 1 })]);
    p.fit(X, y);
    const pred = p.predict(X.slice(0, 2));
    expect(pred[0]).toBeCloseTo(y[0], 2);
    const m = p.params.steps[1];
    const again = MLPRegressor.fromJSON(JSON.parse(JSON.stringify(m)));
    expect(again.predict(X.slice(0, 2))).toEqual(m.predict(X.slice(0, 2)));
    expect(again.summary().layerSizes).toEqual([2, 1]);
  });

  it('accepts the declarative spec', () => {
    const data = X.map(([a, b], i) => ({ a, b, y: y[i] }));
    const m = new MLPRegressor({ layerSizes: [2, 1], optimizer: 'lbfgs', epochs: 50, seed: 1 });
    m.fit({ X: ['a', 'b'], y: 'y', data });
    expect(m.predict({ columns: ['a', 'b'], data: data.slice(0, 1) })[0]).toBeCloseTo(y[0], 2);
  });
});
