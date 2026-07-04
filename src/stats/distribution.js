/**
 * Statistical distributions
 * Provides PDF, CDF, and quantile functions.
 *
 * Since 0.7.0 these delegate to @tangent.to/proba (the tangent suite's
 * scipy.stats-validated distribution package) while preserving the ds
 * parameter conventions ({mean, sd}, {min, max}, {shape, scale}, ...).
 * This replaced the previous approximations (bisection quantiles,
 * normal-approximation tails) with machine-precision implementations.
 */

import { beta as probaBeta, chi2, gamma as probaGamma, normal as probaNormal, uniform as probaUniform } from '@tangent.to/proba';
import { guardPositive, guardProbability } from '../core/math.js';

// ============= Normal Distribution =============

/**
 * Normal distribution
 */
export const normal = {
  /**
   * Probability density function
   * @param {number} x - Value
   * @param {Object} params - {mean, sd}
   * @returns {number} Probability density
   */
  pdf(x, { mean = 0, sd = 1 } = {}) {
    guardPositive(sd, 'sd');
    return probaNormal.pdf(x, { mu: mean, sigma: sd });
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {mean, sd}
   * @returns {number} Cumulative probability
   */
  cdf(x, { mean = 0, sd = 1 } = {}) {
    guardPositive(sd, 'sd');
    return probaNormal.cdf(x, { mu: mean, sigma: sd });
  },

  /**
   * Quantile function (inverse CDF)
   * @param {number} p - Probability
   * @param {Object} params - {mean, sd}
   * @returns {number} Quantile
   */
  quantile(p, { mean = 0, sd = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(sd, 'sd');
    return probaNormal.quantile(p, { mu: mean, sigma: sd });
  }
};

// ============= Uniform Distribution =============

export const uniform = {
  /**
   * Probability density function
   * @param {number} x - Value
   * @param {Object} params - {min, max}
   * @returns {number} Probability density
   */
  pdf(x, { min = 0, max = 1 } = {}) {
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    return probaUniform.pdf(x, { low: min, high: max });
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {min, max}
   * @returns {number} Cumulative probability
   */
  cdf(x, { min = 0, max = 1 } = {}) {
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    return probaUniform.cdf(x, { low: min, high: max });
  },

  /**
   * Quantile function
   * @param {number} p - Probability
   * @param {Object} params - {min, max}
   * @returns {number} Quantile
   */
  quantile(p, { min = 0, max = 1 } = {}) {
    guardProbability(p, 'p');
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    return probaUniform.quantile(p, { low: min, high: max });
  }
};

// ============= Gamma Distribution =============
// ds uses the shape/SCALE convention; proba uses shape/RATE (beta = 1/scale).

export const gamma = {
  /**
   * Probability density function
   * @param {number} x - Value (x > 0)
   * @param {Object} params - {shape, scale}
   * @returns {number} Probability density
   */
  pdf(x, { shape = 1, scale = 1 } = {}) {
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    return probaGamma.pdf(x, { alpha: shape, beta: 1 / scale });
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {shape, scale}
   * @returns {number} Cumulative probability
   */
  cdf(x, { shape = 1, scale = 1 } = {}) {
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    return probaGamma.cdf(x, { alpha: shape, beta: 1 / scale });
  },

  /**
   * Quantile function (inverse CDF)
   * @param {number} p - Probability
   * @param {Object} params - {shape, scale}
   * @returns {number} Quantile
   */
  quantile(p, { shape = 1, scale = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    return probaGamma.quantile(p, { alpha: shape, beta: 1 / scale });
  }
};

// ============= Beta Distribution =============

export const beta = {
  /**
   * Probability density function
   * @param {number} x - Value in [0, 1]
   * @param {Object} params - {alpha, beta}
   * @returns {number} Probability density
   */
  pdf(x, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    return probaBeta.pdf(x, { alpha, beta: betaParam });
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value in [0, 1]
   * @param {Object} params - {alpha, beta}
   * @returns {number} Cumulative probability
   */
  cdf(x, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    return probaBeta.cdf(x, { alpha, beta: betaParam });
  },

  /**
   * Quantile function (inverse CDF)
   * @param {number} p - Probability
   * @param {Object} params - {alpha, beta}
   * @returns {number} Quantile
   */
  quantile(p, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    return probaBeta.quantile(p, { alpha, beta: betaParam });
  }
};

// ============= Chi-square Distribution =============

export const chisq = {
  /**
   * Probability density function
   * @param {number} x - Value (x > 0)
   * @param {Object} params - {df}
   * @returns {number} Probability density
   */
  pdf(x, { df = 1 } = {}) {
    guardPositive(df, 'df');
    return chi2.pdf(x, { k: df });
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {df}
   * @returns {number} Cumulative probability
   */
  cdf(x, { df = 1 } = {}) {
    guardPositive(df, 'df');
    return chi2.cdf(x, { k: df });
  },

  /**
   * Quantile function (inverse CDF)
   * @param {number} p - Probability
   * @param {Object} params - {df}
   * @returns {number} Quantile
   */
  quantile(p, { df = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(df, 'df');
    return chi2.quantile(p, { k: df });
  }
};

/**
 * Chi-square quantile (convenience wrapper kept for backwards compatibility)
 * @param {number} p - Probability
 * @param {number} df - Degrees of freedom
 * @returns {number} Quantile
 */
export function qchisq(p, df) {
  return chisq.quantile(p, { df });
}
