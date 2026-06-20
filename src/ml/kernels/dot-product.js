/**
 * Dot-Product (linear) Kernel
 *
 * k(x1, x2) = sigma0² + x1 · x2
 *
 * A non-stationary kernel: covariance grows with the inputs' inner product
 * (and with distance from the origin). The `sigma0` term is an inhomogeneity /
 * bias controlling the variance of the constant offset. Summed with a Matérn or
 * RBF kernel it adds a global linear trend, as in scikit-learn's
 * `Matern(...) + DotProduct(sigma_0=...) + WhiteKernel(...)`.
 */

import { Kernel } from "./base.js";

export class DotProduct extends Kernel {
  /**
   * @param {number|Object} sigma0OrOpts - Inhomogeneity term, or `{ sigma0 }`.
   * @param {number} [sigma0=1.0]
   *
   * @example
   * new DotProduct(1.0)
   * new DotProduct({ sigma0: 1.0 })
   */
  constructor(sigma0OrOpts = 1.0) {
    super();
    if (typeof sigma0OrOpts === "object" && sigma0OrOpts !== null) {
      this.sigma0 = sigma0OrOpts.sigma0 ?? sigma0OrOpts.sigma_0 ?? 1.0;
    } else {
      this.sigma0 = sigma0OrOpts;
    }
  }

  compute(x1, x2) {
    let dot = 0;
    for (let i = 0; i < x1.length; i++) dot += x1[i] * x2[i];
    return this.sigma0 * this.sigma0 + dot;
  }

  getParams() {
    return { sigma0: this.sigma0 };
  }

  setParams({ sigma0, sigma_0 } = {}) {
    if (sigma0 !== undefined) this.sigma0 = sigma0;
    if (sigma_0 !== undefined) this.sigma0 = sigma_0;
  }
}
