import { describe, it, expect } from 'vitest';
import { RandomForestClassifier, RandomForestRegressor } from '../src/ml/index.js';

describe('RandomForest estimators', () => {
  it('should classify blobs', () => {
    const X = [];
    const y = [];
    const random = () => Math.random();

    const addPoints = (centerX, centerY, label) => {
      for (let i = 0; i < 30; i++) {
        const x = centerX + (random() - 0.5);
        const yVal = centerY + (random() - 0.5);
        X.push([x, yVal]);
        y.push(label);
      }
    };

    addPoints(0, 0, 'A');
    addPoints(5, 5, 'B');

    const forest = new RandomForestClassifier({ nEstimators: 10, maxDepth: 5, seed: 42 });
    forest.fit(X, y);

    const preds = forest.predict([[0.1, -0.2], [5.2, 5.1]]);
    expect(preds).toEqual(['A', 'B']);
  });

  it('should regress smooth function', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 50; i++) {
      const x = -2 + (4 * i) / 49;
      X.push([x]);
      y.push(Math.sin(x));
    }

    const forest = new RandomForestRegressor({ nEstimators: 20, maxDepth: 6, seed: 7 });
    forest.fit(X, y);

    const preds = forest.predict([[0], [Math.PI / 2]]);
    expect(preds[0]).toBeCloseTo(0, 0);
    expect(preds[1]).toBeGreaterThan(0.7);
  });

  describe('Feature Importance', () => {
    it('should compute feature importances for classification', () => {
      const X = [];
      const y = [];

      // Create data where feature 0 is highly predictive
      for (let i = 0; i < 50; i++) {
        X.push([i < 25 ? 0 : 1, Math.random()]);
        y.push(i < 25 ? 'A' : 'B');
      }

      const forest = new RandomForestClassifier({ nEstimators: 10, seed: 42 });
      forest.fit(X, y);

      const importances = forest.featureImportances;
      expect(importances).toBeDefined();
      expect(importances.length).toBe(2);
      expect(importances[0]).toBeGreaterThan(0);
      expect(importances[1]).toBeGreaterThanOrEqual(0);
      // Sum should be 1 (normalized)
      const sum = importances.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });

    it('should compute feature importances for regression', () => {
      const X = [];
      const y = [];

      // Create data where feature 0 is highly predictive
      for (let i = 0; i < 50; i++) {
        const x0 = i / 10;
        const x1 = Math.random();
        X.push([x0, x1]);
        y.push(x0 * 2);  // y depends on x0
      }

      const forest = new RandomForestRegressor({ nEstimators: 10, seed: 42 });
      forest.fit(X, y);

      const importances = forest.featureImportances;
      expect(importances).toBeDefined();
      expect(importances.length).toBe(2);
      expect(importances[0]).toBeGreaterThan(0);
    });
  });

  describe('Out-of-Bag Score', () => {
    it('should compute OOB score for classification', () => {
      const X = [];
      const y = [];

      for (let i = 0; i < 100; i++) {
        const x = i / 100;
        X.push([x, Math.random()]);
        y.push(i < 50 ? 'A' : 'B');
      }

      const forest = new RandomForestClassifier({
        nEstimators: 20,
        oobScore: true,
        seed: 42
      });
      forest.fit(X, y);

      const oobScore = forest.oobScore;
      expect(oobScore).toBeDefined();
      expect(oobScore).toBeGreaterThan(0.5);  // Should be better than random
      expect(oobScore).toBeLessThanOrEqual(1.0);
    });

    it('should compute OOB score for regression', () => {
      const X = [];
      const y = [];

      for (let i = 0; i < 100; i++) {
        const x = i / 100;
        X.push([x]);
        y.push(x * 2 + Math.random() * 0.1);
      }

      const forest = new RandomForestRegressor({
        nEstimators: 20,
        oobScore: true,
        seed: 42
      });
      forest.fit(X, y);

      const oobScore = forest.oobScore;
      expect(oobScore).toBeDefined();
      expect(oobScore).toBeGreaterThan(0.5);  // R^2 should be positive
    });

    it('should throw error when accessing OOB score without oobScore=true', () => {
      const X = [[1, 2], [3, 4]];
      const y = ['A', 'B'];

      const forest = new RandomForestClassifier({ nEstimators: 5 });
      forest.fit(X, y);

      expect(() => forest.oobScore).toThrow();
    });
  });

  describe('Hyperparameters', () => {
    it('should support class_weight="balanced"', () => {
      const X = [];
      const y = [];

      // Imbalanced dataset: 80 A, 20 B
      for (let i = 0; i < 80; i++) {
        X.push([Math.random(), Math.random()]);
        y.push('A');
      }
      for (let i = 0; i < 20; i++) {
        X.push([Math.random() + 5, Math.random() + 5]);
        y.push('B');
      }

      const forest = new RandomForestClassifier({
        nEstimators: 10,
        classWeight: 'balanced',
        seed: 42
      });
      forest.fit(X, y);

      expect(forest.fitted).toBe(true);
      // With balanced weights, the model should be able to predict both classes
      const preds = forest.predict([[5.5, 5.5]]);
      expect(preds[0]).toBe('B');
    });

    it('should support sample_weight', () => {
      const X = [[1], [2], [3], [4]];
      const y = [1, 2, 3, 100];  // Last point is outlier

      // Give very low weight to the outlier
      const sampleWeight = [1, 1, 1, 0.01];

      const forest = new RandomForestRegressor({ nEstimators: 10, seed: 42 });
      forest.fit(X, y, sampleWeight);

      const preds = forest.predict([[5]]);
      // Should predict closer to the trend 1,2,3 rather than being influenced by 100
      expect(preds[0]).toBeLessThan(20);
    });

    it('should support max_samples', () => {
      const X = [];
      const y = [];

      for (let i = 0; i < 100; i++) {
        X.push([i]);
        y.push(i % 2 === 0 ? 'A' : 'B');
      }

      const forest = new RandomForestClassifier({
        nEstimators: 5,
        maxSamples: 50,  // Use only 50 samples per tree
        seed: 42
      });
      forest.fit(X, y);

      expect(forest.fitted).toBe(true);
      const preds = forest.predict([[0], [1]]);
      expect(preds.length).toBe(2);
    });

    it('should support warm_start', () => {
      const X = [[1], [2], [3], [4]];
      const y = ['A', 'A', 'B', 'B'];

      const forest = new RandomForestClassifier({
        nEstimators: 5,
        warmStart: true,
        seed: 42
      });

      forest.fit(X, y);
      expect(forest.forest.trees.length).toBe(5);

      // Fit again with more estimators
      forest.forest.nEstimators = 10;
      forest.fit(X, y);
      expect(forest.forest.trees.length).toBe(10);  // Should have added 5 more
    });

    it('should support min_impurity_decrease', () => {
      const X = [];
      const y = [];

      for (let i = 0; i < 50; i++) {
        X.push([i]);
        y.push(i % 2 === 0 ? 'A' : 'B');
      }

      const forest = new RandomForestClassifier({
        nEstimators: 5,
        minImpurityDecrease: 0.1,  // Require significant impurity decrease
        seed: 42
      });
      forest.fit(X, y);

      expect(forest.fitted).toBe(true);
    });
  });

  describe('Methods', () => {
    it('should support apply() method', () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const y = ['A', 'A', 'B'];

      const forest = new RandomForestClassifier({ nEstimators: 3, seed: 42 });
      forest.fit(X, y);

      const leafIndices = forest.apply([[1, 2], [5, 6]]);
      expect(leafIndices.length).toBe(2);  // 2 samples
      expect(leafIndices[0].length).toBe(3);  // 3 trees
      expect(leafIndices[1].length).toBe(3);
      // Leaf indices should be numbers
      expect(typeof leafIndices[0][0]).toBe('number');
    });

    it('should support decision_path() method', () => {
      const X = [[1, 2], [3, 4]];
      const y = ['A', 'B'];

      const forest = new RandomForestClassifier({ nEstimators: 2, seed: 42 });
      forest.fit(X, y);

      const paths = forest.decisionPath([[1, 2]]);
      expect(paths.length).toBe(1);  // 1 sample
      expect(paths[0].length).toBe(2);  // 2 trees
      expect(Array.isArray(paths[0][0])).toBe(true);  // Each tree has a path
    });
  });
});
