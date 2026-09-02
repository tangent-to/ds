/**
 * White Noise Kernel
 *
 * k(x, x') = noiseLevel · δ(x, x') — independent Gaussian noise on each
 * observation. Unlike every other kernel here it is not a function of the input
 * values: it depends on whether the two arguments are *the same observation*,
 * so two distinct measurements taken at identical inputs are still
 * uncorrelated, and no test point ever shares noise with a training point.
 *
 * This is the learnable counterpart to the regressor's `alpha`. `alpha` is
 * noise you already know (a poll's sampling variance, a sensor's quoted error)
 * and is held fixed; a WhiteKernel is noise you want estimated from the data,
 * and `optimize: true` tunes its `noiseLevel` like any other hyperparameter.
 * scikit-learn draws the same line between `alpha` and `WhiteKernel`.
 *
 * Because the noise is part of the kernel, it enters the predictive variance:
 * `predict(X, { returnStd: true })` returns the standard deviation of a new
 * *observation*, not of the latent function. Noise supplied through `alpha`
 * gives the latent-function std instead. sklearn behaves the same way.
 *
 * @example
 * // Learn the noise level along with the length scale.
 * new GaussianProcessRegressor({
 *   kernel: new SumKernel({ kernels: [new RBF(1.0, 1.0), new WhiteKernel(0.1)] }),
 *   optimize: true,
 * });
 */

import { Kernel } from "./base.js";
import { toMatrix } from "../../core/linalg.js";

export class WhiteKernel extends Kernel {
  /**
   * @param {number|Object} noiseLevelOrOpts - Noise variance, or an options
   *   object `{ noiseLevel, noiseLevelBounds }` (aliases: `noise_level`,
   *   `variance`, `noise_level_bounds`). `noiseLevelBounds` is `[low, high]`,
   *   honoured by hyperparameter optimization; a floor is the usual reason to
   *   set it, since marginal likelihood with many ARD length scales can drive
   *   the noise to zero and explain everything through the kernel.
   *
   * @example
   * new WhiteKernel(0.1)
   * new WhiteKernel({ noiseLevel: 0.1, noiseLevelBounds: [0.05, 2] })
   */
  constructor(noiseLevelOrOpts = 1.0) {
    super();
    if (typeof noiseLevelOrOpts === "object" && noiseLevelOrOpts !== null) {
      const o = noiseLevelOrOpts;
      this.noiseLevel = o.noiseLevel ?? o.noise_level ?? o.variance ?? 1.0;
      // [low, high], honoured by hyperparameter optimization. A floor is the
      // usual reason to set it: marginal likelihood with many ARD length
      // scales can drive the noise to zero and explain everything through
      // the kernel, and a floor at the noise you know is there stops that.
      this.noiseLevelBounds = o.noiseLevelBounds ?? o.noise_level_bounds;
    } else {
      this.noiseLevel = noiseLevelOrOpts;
    }
  }

  /**
   * Covariance between two observations. `noiseLevel` only when they are the
   * same observation — identified by reference, not by value, since the whole
   * point of white noise is that two readings of the same input are still
   * independent. Callers that mean "the variance at this point" pass the same
   * row twice (`compute(x, x)`), which is exactly the diagonal case.
   */
  compute(x1, x2) {
    return x1 === x2 ? this.noiseLevel : 0;
  }

  /**
   * noiseLevel·I for K(X, X), all zeros for a cross-covariance K(X1, X2).
   * Overridden rather than left to the base pointwise loop so the distinction
   * rests on which matrix is being built, not on row identity.
   */
  call(X1, X2 = null) {
    const M1 = toMatrix(X1);
    const n1 = M1.rows;
    const n2 = X2 === null ? n1 : toMatrix(X2).rows;
    const K = new Array(n1);
    for (let i = 0; i < n1; i++) {
      K[i] = new Array(n2).fill(0);
    }
    if (X2 === null) {
      for (let i = 0; i < n1; i++) K[i][i] = this.noiseLevel;
    }
    return toMatrix(K);
  }

  getParams() {
    const p = { noiseLevel: this.noiseLevel };
    if (this.noiseLevelBounds) p.noiseLevelBounds = this.noiseLevelBounds;
    return p;
  }

  setParams({ noiseLevel, noise_level, variance }) {
    const v = noiseLevel ?? noise_level ?? variance;
    if (v !== undefined) this.noiseLevel = v;
  }
}
