/**
 * Constant Kernel
 *
 * Returns a constant covariance independent of the inputs. Useful for
 * representing a constant function prior or combining with other kernels
 * (e.g., ConstantKernel * RBF) to scale amplitudes.
 */

import { Kernel } from "./base.js";

export class ConstantKernel extends Kernel {
  /**
   * @param {number|Object} valueOrOpts - Constant value or options object
   */
  constructor(valueOrOpts = 1.0) {
    super();
    if (typeof valueOrOpts === "object") {
      const { value = 1.0, amplitude } = valueOrOpts;
      this.value = amplitude ?? value;
    } else {
      this.value = valueOrOpts;
    }
  }

  compute() {
    return this.value;
  }

  getParams() {
    return { value: this.value };
  }

  setParams({ value, amplitude }) {
    if (value !== undefined) this.value = value;
    if (amplitude !== undefined) this.value = amplitude;
  }
}
