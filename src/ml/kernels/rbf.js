/**
 * Radial Basis Function (RBF) Kernel
 * 
 * Also known as Squared Exponential or Gaussian kernel.
 * k(x1, x2) = variance * exp(-||x1 - x2||² / (2 * lengthScale²))
 * 
 * Properties:
 * - Infinitely differentiable (very smooth functions)
 * - lengthScale controls how quickly correlation decays with distance
 * - variance (amplitude) controls the amplitude of the function
 */

import { Kernel } from './base.js';

export class RBF extends Kernel {
  /**
   * @param {number|Object} lengthScaleOrOpts - Length scale or options object
   * @param {number} [variance] - Signal variance (default: 1.0)
   * 
   * @example
   * // Positional arguments (scikit-learn style)
   * new RBF(1.0, 1.0)
   * 
   * @example
   * // Object arguments
   * new RBF({ lengthScale: 1.0, amplitude: 1.0 })
   */
  constructor(lengthScaleOrOpts = 1.0, variance = 1.0) {
    super();
    
    if (typeof lengthScaleOrOpts === 'object') {
      // Object-style constructor
      this.lengthScale = lengthScaleOrOpts.lengthScale ?? lengthScaleOrOpts.length_scale ?? 1.0;
      this.variance = lengthScaleOrOpts.variance ?? lengthScaleOrOpts.amplitude ?? 1.0;
    } else {
      // Positional arguments
      this.lengthScale = lengthScaleOrOpts;
      this.variance = variance;
    }
  }

  compute(x1, x2) {
    // Length scale may be scalar (isotropic) or a per-dimension array (ARD).
    const l = this.lengthScale;
    const isArr = Array.isArray(l);
    let scaledSq = 0;
    for (let i = 0; i < x1.length; i++) {
      const li = isArr ? l[i] : l;
      const s = (x1[i] - x2[i]) / li;
      scaledSq += s * s;
    }
    return this.variance * Math.exp(-scaledSq / 2);
  }

  getParams() {
    return {
      lengthScale: this.lengthScale,
      variance: this.variance
    };
  }

  setParams({ lengthScale, variance, amplitude }) {
    if (lengthScale !== undefined) this.lengthScale = lengthScale;
    if (variance !== undefined) this.variance = variance;
    if (amplitude !== undefined) this.variance = amplitude;
  }
}
