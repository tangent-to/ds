import { describe, it, expect } from 'vitest';
import {
  mseLoss,
  maeLoss,
  logLoss,
  crossEntropy,
  hingeLoss,
  huberLoss,
  getLossFunction
} from '../src/ml/loss.js';

describe('Loss Functions', () => {
  describe('mseLoss', () => {
    it('should compute MSE loss correctly', () => {
      const yTrue = [1, 2, 3, 4];
      const yPred = [1.1, 2.1, 2.9, 4.2];

      const loss = mseLoss(yTrue, yPred);

      // MSE = mean((yPred - yTrue)^2)
      const expectedLoss = ((0.1**2 + 0.1**2 + 0.1**2 + 0.2**2) / 4);
      expect(loss).toBeCloseTo(expectedLoss, 6);

      // Gradient should have same length as inputs
    });

    it('should have zero loss for perfect predictions', () => {
      const yTrue = [1, 2, 3];
      const yPred = [1, 2, 3];

      const loss = mseLoss(yTrue, yPred);

      expect(loss).toBeCloseTo(0, 10);
    });
  });

  describe('maeLoss', () => {
    it('should compute MAE loss correctly', () => {
      const yTrue = [1, 2, 3, 4];
      const yPred = [1.5, 2.5, 2.5, 4.5];

      const loss = maeLoss(yTrue, yPred);

      // MAE = mean(|yPred - yTrue|)
      const expectedLoss = (0.5 + 0.5 + 0.5 + 0.5) / 4;
      expect(loss).toBeCloseTo(expectedLoss, 6);

    });

    it('should handle negative errors', () => {
      const yTrue = [5];
      const yPred = [3];

      const loss = maeLoss(yTrue, yPred);

      expect(loss).toBeCloseTo(2, 6);
    });
  });

  describe('logLoss', () => {
    it('should compute binary cross-entropy correctly', () => {
      const yTrue = [1, 0, 1, 0];
      const yPred = [0.9, 0.1, 0.8, 0.2];

      const loss = logLoss(yTrue, yPred);

      // Loss should be positive
      expect(loss).toBeGreaterThan(0);
      expect(loss).toBeLessThan(1);

    });

    it('should penalize confident wrong predictions', () => {
      const yTrue = [1, 1];
      const yPred1 = [0.9, 0.9]; // Good predictions
      const yPred2 = [0.1, 0.1]; // Bad predictions

      const loss1 = logLoss(yTrue, yPred1);
      const loss2 = logLoss(yTrue, yPred2);

      expect(loss2).toBeGreaterThan(loss1);
    });

    it('should clip extreme predictions', () => {
      const yTrue = [1, 0];
      const yPred = [1.0, 0.0]; // Would cause log(0)

      const loss = logLoss(yTrue, yPred);

      // Should not be infinite
      expect(isFinite(loss)).toBe(true);
    });
  });

  describe('crossEntropy', () => {
    it('should compute categorical cross-entropy', () => {
      const yTrue = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];
      const yPred = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
      ];

      const loss = crossEntropy(yTrue, yPred);

      expect(loss).toBeGreaterThan(0);
    });

    it('should have low loss for good predictions', () => {
      const yTrue = [
        [1, 0, 0],
        [0, 1, 0]
      ];
      const yPred = [
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025]
      ];

      const loss = crossEntropy(yTrue, yPred);

      expect(loss).toBeLessThan(0.2);
    });
  });

  describe('hingeLoss', () => {
    it('should compute hinge loss for SVM', () => {
      const yTrue = [1, 1, -1, -1];
      const yPred = [2, 0.5, -2, -0.5];

      const loss = hingeLoss(yTrue, yPred);

      expect(loss).toBeGreaterThanOrEqual(0);
    });

    it('should have zero loss for correct classifications', () => {
      const yTrue = [1, -1];
      const yPred = [2, -2];

      const loss = hingeLoss(yTrue, yPred);

      expect(loss).toBeCloseTo(0, 6);
    });

    it('should penalize misclassifications', () => {
      const yTrue = [1, 1];
      const yPred = [-1, -1]; // Wrong

      const loss = hingeLoss(yTrue, yPred);

      expect(loss).toBeGreaterThan(1);
    });
  });

  describe('huberLoss', () => {
    it('should behave like MSE for small errors', () => {
      const yTrue = [1, 2, 3];
      const yPred = [1.1, 2.1, 3.1];

      const loss = huberLoss(yTrue, yPred, 1.0);

      // Small errors, should be quadratic
      expect(loss).toBeGreaterThan(0);
      expect(loss).toBeLessThan(0.1);
    });

    it('should behave like MAE for large errors', () => {
      const yTrue = [0];
      const yPred = [5];

      const loss = huberLoss(yTrue, yPred, 1.0);

      // Large error, should be linear
      expect(loss).toBeGreaterThan(0);
    });

    it('should handle different delta values', () => {
      const yTrue = [0, 0];
      const yPred = [2, 2];

      const loss1 = huberLoss(yTrue, yPred, 0.5);
      const loss2 = huberLoss(yTrue, yPred, 2.0);

      // Both should be finite
      expect(isFinite(loss1)).toBe(true);
      expect(isFinite(loss2)).toBe(true);
    });
  });

  describe('getLossFunction', () => {
    it('should return MSE by name', () => {
      const lossFn1 = getLossFunction('mse');
      const lossFn2 = getLossFunction('mean_squared_error');

      expect(lossFn1).toBe(mseLoss);
      expect(lossFn2).toBe(mseLoss);
    });

    it('should return MAE by name', () => {
      const lossFn = getLossFunction('mae');
      expect(lossFn).toBe(maeLoss);
    });

    it('should return log loss by name', () => {
      const lossFn1 = getLossFunction('log');
      const lossFn2 = getLossFunction('binary_crossentropy');

      expect(lossFn1).toBe(logLoss);
      expect(lossFn2).toBe(logLoss);
    });

    it('should throw error for unknown loss', () => {
      expect(() => {
        getLossFunction('unknown_loss');
      }).toThrow('Unknown loss function');
    });
  });
});
