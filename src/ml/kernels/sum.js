/**
 * Sum Kernel
 *
 * Combines multiple kernels by summing their covariance values. Enables
 * additive compositions such as RBF + RationalQuadratic, or adding a
 * ConstantKernel to raise the baseline variance.
 */

import { Kernel } from "./base.js";

export class SumKernel extends Kernel {
  /**
   * @param {Object} opts
   * @param {Kernel[]} opts.kernels - Array of kernel instances to sum
   */
  constructor(opts = {}) {
    super();
    const { kernels = [] } = opts;
    if (!Array.isArray(kernels) || kernels.length === 0) {
      throw new Error("SumKernel requires a non-empty array of kernels");
    }
    kernels.forEach((k, idx) => {
      if (!(k instanceof Kernel)) {
        throw new Error(`SumKernel expects Kernel instances (index ${idx})`);
      }
    });
    this.kernels = kernels;
  }

  compute(x1, x2) {
    return this.kernels.reduce((sum, kernel) => sum + kernel.compute(x1, x2), 0);
  }

  /**
   * Sum the children's covariance MATRICES rather than their pointwise
   * `compute()` values. Identical numbers for kernels that are plain functions
   * of the input values, but a WhiteKernel is not one: it must know whether the
   * matrix being built is K(X, X) or a cross-covariance K(X1, X2), which only
   * `call()` can tell it. Delegating per element would silently drop the noise
   * term (or, worse, leak it into the train/test block).
   */
  call(X1, X2 = null) {
    const first = this.kernels[0].call(X1, X2);
    for (let k = 1; k < this.kernels.length; k++) {
      const Kk = this.kernels[k].call(X1, X2);
      for (let i = 0; i < first.rows; i++) {
        for (let j = 0; j < first.columns; j++) {
          first.set(i, j, first.get(i, j) + Kk.get(i, j));
        }
      }
    }
    return first;
  }

  getParams() {
    return {
      kernels: this.kernels.map((kernel) => ({
        type: kernel.constructor.name,
        params: kernel.getParams(),
      })),
    };
  }

  setParams({ kernels }) {
    if (kernels) {
      throw new Error("SumKernel.setParams() does not support replacing child kernels");
    }
  }
}
