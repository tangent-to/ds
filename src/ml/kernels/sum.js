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
