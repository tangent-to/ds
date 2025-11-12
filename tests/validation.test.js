import { beforeEach, describe, expect, it } from 'vitest';
import { prepareXY } from '../src/core/table.js';
import { StandardScaler } from '../src/ml/preprocessing.js';
import {
  crossValidate,
  groupKFold,
  kFold,
  leaveOneOut,
  shuffleSplit,
  stratifiedKFold,
  trainTestSplit,
} from '../src/ml/validation.js';
import { setSeed } from '../src/ml/utils.js';

describe('ML Validation', () => {
  beforeEach(() => {
    setSeed(42); // Ensure reproducibility
  });

  describe('trainTestSplit', () => {
    it('should split data with default ratio', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];

      const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { shuffle: false });

      expect(XTrain.length).toBe(4);
      expect(XTest.length).toBe(1);
      expect(yTrain.length).toBe(4);
      expect(yTest.length).toBe(1);
    });

    it('should respect custom ratio', () => {
      const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
      const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const { XTrain, XTest } = trainTestSplit(X, y, { ratio: 0.7, shuffle: false });

      expect(XTrain.length).toBe(7);
      expect(XTest.length).toBe(3);
    });

    it('should work without y', () => {
      const X = [[1], [2], [3]];
      const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X);

      expect(XTrain.length).toBeGreaterThan(0);
      expect(XTest.length).toBeGreaterThan(0);
      expect(yTrain).toBeUndefined();
      expect(yTest).toBeUndefined();
    });

    it('should be reproducible with seed', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];

      const split1 = trainTestSplit(X, y, { seed: 123 });
      const split2 = trainTestSplit(X, y, { seed: 123 });

      expect(split1.trainIndices).toEqual(split2.trainIndices);
      expect(split1.testIndices).toEqual(split2.testIndices);
    });

    it('should split declarative table inputs and expose table views', () => {
      const data = [
        { x1: 1, x2: 2, y: 0 },
        { x1: 3, x2: 4, y: 1 },
        { x1: 5, x2: 6, y: 0 },
        { x1: 7, x2: 8, y: 1 },
      ];

      const split = trainTestSplit(
        { data, X: ['x1', 'x2'], y: 'y' },
        { ratio: 0.5, shuffle: false },
      );

      expect(split.columns).toEqual(['x1', 'x2']);
      expect(split.train.data).toHaveLength(2);
      expect(split.test.data).toHaveLength(2);
      expect(split.train.y).toEqual(split.yTrain);
      expect(split.train.indices).toEqual(split.trainIndices);
      expect(split.train.data[0].y).toBe(0);
    });
  });

  describe('encoder reuse with table descriptors', () => {
    it('should reuse label encoder metadata from trainTestSplit', () => {
      const data = [
        { value: 1, label: 'A' },
        { value: 2, label: 'B' },
        { value: 3, label: 'C' },
        { value: 4, label: 'A' },
      ];

      const split = trainTestSplit(
        { data, X: ['value'], y: 'label' },
        { ratio: 0.5, shuffle: false },
      );

      expect(split.train.metadata).toBeDefined();
      expect(split.train.metadata.encoders?.y).toBeDefined();

      const prepared = prepareXY({
        data: split.train.data,
        X: ['value'],
        y: 'label',
        encoders: split.train.metadata.encoders,
      });

      expect(prepared.y).toEqual(split.yTrain);

      const scaler = new StandardScaler().fit({
        data: split.train.data,
        columns: ['value'],
      });

      const scaledTrain = scaler.transform({
        data: split.train.data,
        columns: ['value'],
        encoders: split.train.metadata.encoders,
      });

      const preparedScaled = prepareXY({
        data: scaledTrain.data,
        X: ['value'],
        y: 'label',
        encoders: scaledTrain.metadata?.encoders,
      });

      expect(preparedScaled.y).toEqual(split.yTrain);
    });
  });

  describe('kFold', () => {
    it('should create k folds', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];

      const folds = kFold(X, y, 5);

      expect(folds.length).toBe(5);
      expect(folds[0].description).toBe('KFold');
      expect(folds[0].nSplits).toBe(5);
    });

    it('should have non-overlapping test sets', () => {
      const X = [[1], [2], [3], [4]];
      const y = [1, 2, 3, 4];

      const folds = kFold(X, y, 2, false);

      const allTestIndices = folds.flatMap((f) => f.testIndices);
      const uniqueTestIndices = new Set(allTestIndices);

      expect(uniqueTestIndices.size).toBe(X.length);
    });

    it('should throw error for invalid k', () => {
      const X = [[1], [2]];
      const y = [1, 2];

      expect(() => kFold(X, y, 1)).toThrow();
      expect(() => kFold(X, y, 3)).toThrow();
    });

    it('should support table descriptors with table views', () => {
      const data = [
        { x: 1, y: 0 },
        { x: 2, y: 1 },
        { x: 3, y: 0 },
        { x: 4, y: 1 },
      ];

      const folds = kFold(
        { data, X: ['x'], y: 'y' },
        { k: 2, shuffle: false },
      );

      expect(folds).toHaveLength(2);
      expect(folds[0].train.data.length).toBe(2);
      expect(folds[0].test.data.length).toBe(2);
      expect(folds[0].train.columns).toEqual(['x']);
      expect(folds[0].train.y.every((val) => typeof val === 'number')).toBe(true);
      expect(folds[0].test.data[0]).toHaveProperty('y');
      expect(folds[0].train.metadata).toBeDefined();
      expect(folds[0].train.metadata.columns).toEqual(['x']);
    });
  });

  describe('stratifiedKFold', () => {
    it('should maintain class proportions', () => {
      const X = [[1], [2], [3], [4], [5], [6]];
      const y = ['A', 'A', 'A', 'B', 'B', 'B'];

      const folds = stratifiedKFold(X, y, 2);

      expect(folds.length).toBe(2);

      // Each fold should have samples from both classes
      folds.forEach((fold) => {
        const testLabels = fold.testIndices.map((i) => y[i]);
        const hasA = testLabels.includes('A');
        const hasB = testLabels.includes('B');
        expect(hasA).toBe(true);
        expect(hasB).toBe(true);
      });
    });

    it('should accept table descriptors with declared y', () => {
      const data = [
        { f1: 1, label: 'A' },
        { f1: 2, label: 'A' },
        { f1: 3, label: 'B' },
        { f1: 4, label: 'B' },
      ];

      const folds = stratifiedKFold(
        { data, X: ['f1'], y: 'label' },
        { k: 2 },
      );

      expect(folds).toHaveLength(2);
      folds.forEach((fold) => {
        expect(fold.train.data.length).toBe(2);
        const trainLabels = fold.train.y;
        expect(new Set(trainLabels).size).toBe(2); // both classes present
        expect(fold.train.columns).toEqual(['f1']);
      });
    });
  });

  describe('groupKFold', () => {
    it('should keep groups together', () => {
      const X = [[1], [2], [3], [4]];
      const y = [1, 2, 3, 4];
      const groups = ['A', 'A', 'B', 'B'];

      const folds = groupKFold(X, y, groups, 2);

      expect(folds.length).toBe(2);

      // Check that groups are not split
      folds.forEach((fold) => {
        const testGroups = fold.testIndices.map((i) => groups[i]);
        const uniqueTestGroups = new Set(testGroups);

        fold.trainIndices.forEach((i) => {
          expect(uniqueTestGroups.has(groups[i])).toBe(false);
        });
      });
    });

    it('should throw error if k > number of groups', () => {
      const X = [[1], [2]];
      const y = [1, 2];
      const groups = ['A', 'A'];

      expect(() => groupKFold(X, y, groups, 3)).toThrow();
    });

    it('should work with descriptor input referencing group column', () => {
      const data = [
        { x: 1, group: 'A' },
        { x: 2, group: 'A' },
        { x: 3, group: 'B' },
        { x: 4, group: 'B' },
      ];

      const folds = groupKFold(
        { data, X: ['x'], groups: 'group' },
        { k: 2 },
      );

      expect(folds.length).toBe(2);
      folds.forEach((fold) => {
        expect(fold.train.data.length + fold.test.data.length).toBe(4);
        expect(new Set(fold.test.data.map((row) => row.group)).size).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('leaveOneOut', () => {
    it('should create n folds for n samples', () => {
      const X = [[1], [2], [3]];
      const y = [1, 2, 3];

      const folds = leaveOneOut(X, y);

      expect(folds.length).toBe(3);
      expect(folds[0].description).toBe('LeaveOneOut');
    });

    it('should have single test sample per fold', () => {
      const X = [[1], [2], [3]];
      const y = [1, 2, 3];

      const folds = leaveOneOut(X, y);

      folds.forEach((fold) => {
        expect(fold.testIndices.length).toBe(1);
        expect(fold.trainIndices.length).toBe(2);
      });
    });

    it('should return table-aware folds when given descriptor', () => {
      const data = [
        { x: 1, y: 10 },
        { x: 2, y: 20 },
        { x: 3, y: 30 },
      ];

      const folds = leaveOneOut({ data, X: ['x'], y: 'y' });
      expect(folds).toHaveLength(3);
      expect(folds[0].train.data.length).toBe(2);
      expect(folds[0].train.y.length).toBe(2);
      expect(folds[0].train.columns).toEqual(['x']);
    });
  });

  describe('shuffleSplit', () => {
    it('should create n splits', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];

      const splits = shuffleSplit(X, y, { nSplits: 3 });

      expect(splits.length).toBe(3);
      expect(splits[0].description).toBe('ShuffleSplit');
    });

    it('should respect test ratio', () => {
      const X = Array.from({ length: 10 }, (_, i) => [i]);
      const y = Array.from({ length: 10 }, (_, i) => i);

      const splits = shuffleSplit(X, y, { nSplits: 2, testRatio: 0.3 });

      splits.forEach((split) => {
        expect(split.testIndices.length).toBe(3);
        expect(split.trainIndices.length).toBe(7);
      });
    });

    it('should support table descriptors and expose table views', () => {
      const data = [
        { value: 1 },
        { value: 2 },
        { value: 3 },
        { value: 4 },
      ];

      const splits = shuffleSplit(
        { data, X: ['value'] },
        { nSplits: 1, testRatio: 0.25, seed: 42 },
      );

      expect(splits).toHaveLength(1);
      expect(splits[0].train.data.length).toBe(3);
      expect(splits[0].test.data.length).toBe(1);
      expect(splits[0].train.columns).toEqual(['value']);
    });
  });

  describe('crossValidate', () => {
    it('should perform cross-validation', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];

      const fitFn = (XTrain, yTrain) => {
        // Simple mean model
        const meanY = yTrain.reduce((a, b) => a + b, 0) / yTrain.length;
        return { predict: () => [meanY, meanY] };
      };

      const scoreFn = (model, XTest, yTest) => {
        const predictions = model.predict();
        // Simple accuracy (within 2 of true value)
        let correct = 0;
        for (let i = 0; i < yTest.length; i++) {
          if (Math.abs(predictions[i] - yTest[i]) < 2) correct++;
        }
        return correct / yTest.length;
      };

      const folds = kFold(X, y, 2, false);
      const result = crossValidate(fitFn, scoreFn, X, y, folds);

      expect(result.scores.length).toBe(2);
      expect(result.meanScore).toBeGreaterThanOrEqual(0);
      expect(result.meanScore).toBeLessThanOrEqual(1);
      expect(result.stdScore).toBeGreaterThanOrEqual(0);
    });

    it('should accept table descriptors without explicit folds', () => {
      const data = [
        { feature: 1, target: 1 },
        { feature: 2, target: 2 },
        { feature: 3, target: 3 },
        { feature: 4, target: 4 },
      ];

      const fitFn = (XTrain, yTrain) => {
        const meanY = yTrain.reduce((sum, v) => sum + v, 0) / yTrain.length;
        return {
          predict: (XTest) => XTest.map(() => meanY),
        };
      };

      const scoreFn = (model, XTest, yTest) => {
        const preds = model.predict(XTest);
        let error = 0;
        for (let i = 0; i < preds.length; i++) {
          error += Math.abs(preds[i] - yTest[i]);
        }
        return -error;
      };

      const result = crossValidate(
        fitFn,
        scoreFn,
        { data, X: ['feature'], y: 'target' },
        { k: 2 },
      );

      expect(result.scores.length).toBe(2);
      expect(result.tableFolds).toHaveLength(2);
      expect(result.metadata.columns).toEqual(['feature']);
    });
  });
});
