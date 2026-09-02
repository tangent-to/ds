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

import { Kernel, checkBlocks } from './base.js';

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
  /**
   * @param {number|number[]|Object} [lengthScaleOrOpts=1] - a length scale, one
   *   per input dimension (ARD), or an options object with `lengthScale`,
   *   `variance`, and optionally `blocks` (ARD by block, see Matern),
   *   `lengthScaleBounds` and `varianceBounds` (`[low, high]`, honoured by
   *   hyperparameter optimization)
   * @param {number} [variance=1]
   */
  constructor(lengthScaleOrOpts = 1.0, variance = 1.0) {
    super();
    
    // An ARD length scale is an ARRAY, and `typeof [] === 'object'`, so the
    // options branch would swallow it: `new RBF([1.3, 0.7], 0.8)` used to
    // find no .lengthScale on the array and silently fall back to the
    // defaults, discarding BOTH arguments. Arrays take the positional path.
    if (typeof lengthScaleOrOpts === 'object' && !Array.isArray(lengthScaleOrOpts)) {
      // Object-style constructor
      this.lengthScale = lengthScaleOrOpts.lengthScale ?? lengthScaleOrOpts.length_scale ?? 1.0;
      this.variance = lengthScaleOrOpts.variance ?? lengthScaleOrOpts.amplitude ?? 1.0;
      // ARD by block and optional bounds; see Matern for the reasoning.
      this.blocks = lengthScaleOrOpts.blocks;
      this.lengthScaleBounds = lengthScaleOrOpts.lengthScaleBounds;
      this.varianceBounds = lengthScaleOrOpts.varianceBounds;
    } else {
      // Positional arguments
      this.lengthScale = lengthScaleOrOpts;
      this.variance = variance;
    }
    checkBlocks(this, 'RBF');
  }

  compute(x1, x2) {
    // Length scale may be scalar (isotropic) or a per-dimension array (ARD).
    const l = this.lengthScale;
    const isArr = Array.isArray(l);
    const blocks = this.blocks;
    let scaledSq = 0;
    for (let i = 0; i < x1.length; i++) {
      const li = isArr ? l[blocks ? blocks[i] : i] : l;
      const s = (x1[i] - x2[i]) / li;
      scaledSq += s * s;
    }
    return this.variance * Math.exp(-scaledSq / 2);
  }

  getParams() {
    const p = { lengthScale: this.lengthScale, variance: this.variance };
    if (this.blocks) p.blocks = this.blocks;
    if (this.lengthScaleBounds) p.lengthScaleBounds = this.lengthScaleBounds;
    if (this.varianceBounds) p.varianceBounds = this.varianceBounds;
    return p;
  }

  setParams({ lengthScale, variance, amplitude }) {
    if (lengthScale !== undefined) this.lengthScale = lengthScale;
    if (variance !== undefined) this.variance = variance;
    if (amplitude !== undefined) this.variance = amplitude;
  }
}
