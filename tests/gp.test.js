/**
 * Gaussian Process Regressor and Kernels tests
 *
 * Testing strategies for stochastic models:
 * 1. Deterministic kernel properties (symmetry, self-covariance)
 * 2. Interpolation (low noise → predictions match training data)
 * 3. Uncertainty quantification (std ≈ 0 at training points)
 * 4. Seed reproducibility
 * 5. Statistical properties over many samples
 */

import { describe, it, expect } from 'vitest';
import {
  GaussianProcessRegressor,
  RBF,
  Matern,
  Periodic,
  RationalQuadratic,
  ConstantKernel,
  SumKernel,
  Kernel
} from '../src/ml/index.js';

describe('Kernels', () => {
  describe('RBF Kernel', () => {
    it('should compute k(x,x) = variance', () => {
      const rbf = new RBF(1.0, 2.0); // lengthScale=1, variance=2
      expect(rbf.compute([0], [0])).toBeCloseTo(2.0);
      expect(rbf.compute([5, 3], [5, 3])).toBeCloseTo(2.0);
    });

    it('should be symmetric: k(x,y) = k(y,x)', () => {
      const rbf = new RBF(1.0, 1.0);
      const k1 = rbf.compute([1, 2], [3, 4]);
      const k2 = rbf.compute([3, 4], [1, 2]);
      expect(k1).toBeCloseTo(k2);
    });

    it('should decay with distance', () => {
      const rbf = new RBF(1.0, 1.0);
      const k_close = rbf.compute([0], [0.1]);
      const k_far = rbf.compute([0], [5]);
      expect(k_close).toBeGreaterThan(k_far);
    });

    it('should accept object-style construction', () => {
      const rbf = new RBF({ lengthScale: 2.0, amplitude: 3.0 });
      expect(rbf.getParams()).toEqual({ lengthScale: 2.0, variance: 3.0 });
    });

    it('should compute covariance matrix with call()', () => {
      const rbf = new RBF(1.0, 1.0);
      const X = [[0], [1], [2]];
      const K = rbf.call(X, X);
      expect(K.rows).toBe(3);
      expect(K.columns).toBe(3);
      // Diagonal should be variance
      expect(K.get(0, 0)).toBeCloseTo(1.0);
      expect(K.get(1, 1)).toBeCloseTo(1.0);
      // Should be symmetric
      expect(K.get(0, 1)).toBeCloseTo(K.get(1, 0));
    });
  });

  describe('Matern Kernel', () => {
    it('should compute k(x,x) = amplitude', () => {
      const matern = new Matern({ lengthScale: 1.0, nu: 1.5, amplitude: 2.0 });
      expect(matern.compute([0], [0])).toBeCloseTo(2.0);
    });

    it('should support nu = 0.5, 1.5, 2.5, Infinity', () => {
      const nus = [0.5, 1.5, 2.5, Infinity];
      for (const nu of nus) {
        const m = new Matern({ lengthScale: 1.0, nu, amplitude: 1.0 });
        expect(m.compute([0], [0])).toBeCloseTo(1.0);
        expect(m.compute([0], [1])).toBeLessThan(1.0);
      }
    });

    it('should equal RBF when nu = Infinity', () => {
      const matern = new Matern({ lengthScale: 1.0, nu: Infinity, amplitude: 1.0 });
      const rbf = new RBF(1.0, 1.0);
      expect(matern.compute([0], [1])).toBeCloseTo(rbf.compute([0], [1]), 5);
    });
  });

  describe('Periodic Kernel', () => {
    it('should have period-repeating covariance', () => {
      const period = 2.0;
      const periodic = new Periodic(1.0, period, 1.0);
      // k(0, period) should equal k(0, 0) = variance
      expect(periodic.compute([0], [period])).toBeCloseTo(1.0, 1);
      expect(periodic.compute([0], [2 * period])).toBeCloseTo(1.0, 1);
    });
  });

  describe('RationalQuadratic Kernel', () => {
    it('should compute k(x,x) = variance', () => {
      const rq = new RationalQuadratic(1.0, 1.0, 2.0);
      expect(rq.compute([0], [0])).toBeCloseTo(2.0);
    });

    it('should accept object-style construction', () => {
      const rq = new RationalQuadratic({ lengthScale: 1.0, alpha: 0.5, amplitude: 1.0 });
      expect(rq.getParams().lengthScale).toBe(1.0);
      expect(rq.getParams().alpha).toBe(0.5);
    });
  });

  describe('ConstantKernel', () => {
    it('should return constant value everywhere', () => {
      const c = new ConstantKernel({ value: 5.0 });
      expect(c.compute([0], [0])).toBe(5.0);
      expect(c.compute([1, 2, 3], [100, 200, 300])).toBe(5.0);
    });
  });

  describe('SumKernel', () => {
    it('should sum component kernels', () => {
      const c = new ConstantKernel({ value: 2.0 });
      const rbf = new RBF(1.0, 1.0);
      const sum = new SumKernel({ kernels: [c, rbf] });

      // k(x,x) should be constant + rbf_variance = 2 + 1 = 3
      expect(sum.compute([0], [0])).toBeCloseTo(3.0);
    });
  });
});

describe('GaussianProcessRegressor', () => {
  describe('Construction', () => {
    it('should create GP with string kernel type', () => {
      const gp = new GaussianProcessRegressor({ kernel: 'rbf', lengthScale: 1.0 });
      expect(gp.kernel).toBeInstanceOf(RBF);
    });

    it('should create GP with kernel instance', () => {
      const kernel = new Matern({ lengthScale: 1.0, nu: 2.5 });
      const gp = new GaussianProcessRegressor({ kernel });
      expect(gp.kernel).toBeInstanceOf(Matern);
    });

    it('should accept noiseLevel as alias for alpha', () => {
      const gp = new GaussianProcessRegressor({ kernel: 'rbf', noiseLevel: 0.1 });
      expect(gp.alpha).toBe(0.1);
    });
  });

  describe('Interpolation (low noise)', () => {
    it('should interpolate training points exactly with very low noise', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-10
      });

      const X_train = [[0], [1], [2], [3], [4]];
      const y_train = [0, 1, 0.5, 0.8, 0.2];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 3);
      }
    });
  });

  describe('Uncertainty Quantification', () => {
    it('should have low std at training points', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-5
      });

      const X_train = [[0], [2], [4]];
      const y_train = [0, 1, 0];

      gp.fit(X_train, y_train);
      const { mean, std } = gp.predict(X_train, { returnStd: true });

      // Std at training points should be very small
      for (const s of std) {
        expect(s).toBeLessThan(0.01);
      }
    });

    it('should have higher std away from training points', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        variance: 1.0,
        alpha: 1e-5
      });

      const X_train = [[0], [4]];
      const y_train = [0, 0];

      gp.fit(X_train, y_train);

      const { std: std_at_train } = gp.predict([[0]], { returnStd: true });
      const { std: std_away } = gp.predict([[2]], { returnStd: true });

      expect(std_away[0]).toBeGreaterThan(std_at_train[0]);
    });

    it('should return full covariance matrix with returnCov', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-5
      });

      gp.fit([[0], [1]], [0, 1]);
      const { mean, covariance } = gp.predict([[0], [0.5], [1]], { returnCov: true });

      expect(mean.length).toBe(3);
      // covariance is returned as 2D array
      expect(covariance.length).toBe(3);
      expect(covariance[0].length).toBe(3);
      // Covariance matrix should be symmetric
      expect(covariance[0][1]).toBeCloseTo(covariance[1][0]);
    });
  });

  describe('Sampling', () => {
    it('should produce reproducible samples with seed', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 0.1
      });

      gp.fit([[0], [1], [2]], [0, 1, 0]);

      const X_test = [[0], [0.5], [1], [1.5], [2]];
      const samples1 = gp.sample(X_test, 3, 42);
      const samples2 = gp.sample(X_test, 3, 42);

      expect(samples1).toEqual(samples2);
    });

    it('should produce different samples with different seeds', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 0.1
      });

      gp.fit([[0], [1]], [0, 1]);

      const X_test = [[0.5]];
      const samples1 = gp.sample(X_test, 1, 1);
      const samples2 = gp.sample(X_test, 1, 999);

      // Different seeds should give different samples
      expect(samples1[0][0]).not.toBe(samples2[0][0]);
    });

    it('should sample from prior (before fitting)', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        variance: 1.0
      });

      const X = [[0], [1], [2], [3], [4]];
      const samples = gp.samplePrior(X, 5, 123);

      expect(samples.length).toBe(5);
      expect(samples[0].length).toBe(5);
    });

    it('sample mean should converge to predicted mean', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 0.1
      });

      gp.fit([[0], [2]], [0, 2]);

      const X_test = [[1]];
      const predicted = gp.predict(X_test)[0];

      // Draw many samples and compute mean
      const nSamples = 200;
      let sum = 0;
      for (let i = 0; i < nSamples; i++) {
        const sample = gp.sample(X_test, 1, i * 1000);  // Use spread-out seeds
        sum += sample[0][0];
      }
      const sampleMean = sum / nSamples;

      // Sample mean should be close to predicted mean
      // Allow larger tolerance due to stochastic nature
      expect(Math.abs(sampleMean - predicted)).toBeLessThan(0.5);
    });
  });

  describe('Different Kernels', () => {
    it('should work with rbf kernel', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-5
      });

      const X_train = [[0], [1], [2]];
      const y_train = [0, 1, 0];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 2);
      }
    });

    it('should work with matern kernel', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'matern',
        lengthScale: 1.0,
        nu: 1.5,
        alpha: 1e-5
      });

      const X_train = [[0], [1], [2]];
      const y_train = [0, 1, 0];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 2);
      }
    });

    it('should work with periodic kernel', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'periodic',
        lengthScale: 1.0,
        period: 2.0,
        alpha: 1e-5
      });

      const X_train = [[0], [1], [2]];
      const y_train = [0, 1, 0];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 2);
      }
    });

    it('should work with rationalquadratic kernel (using kernel instance)', () => {
      // Use kernel instance to avoid alpha parameter conflict
      const kernel = new RationalQuadratic(1.0, 1.0, 1.0);
      const gp = new GaussianProcessRegressor({
        kernel,
        alpha: 1e-5
      });

      const X_train = [[0], [1], [2]];
      const y_train = [0, 1, 0];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 2);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle single training point', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-5
      });

      gp.fit([[0]], [1]);
      const pred = gp.predict([[0]]);
      expect(pred[0]).toBeCloseTo(1, 3);
    });

    it('should handle multidimensional input', () => {
      const gp = new GaussianProcessRegressor({
        kernel: 'rbf',
        lengthScale: 1.0,
        alpha: 1e-5
      });

      const X_train = [[0, 0], [1, 0], [0, 1], [1, 1]];
      const y_train = [0, 1, 1, 2];

      gp.fit(X_train, y_train);
      const y_pred = gp.predict(X_train);

      for (let i = 0; i < y_train.length; i++) {
        expect(y_pred[i]).toBeCloseTo(y_train[i], 2);
      }
    });
  });
});
