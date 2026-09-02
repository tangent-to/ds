/**
 * Matérn Kernel
 *
 * Supports ν = 0.5, 1.5, 2.5, or Infinity (which reduces to the RBF kernel).
 * These common cases cover most GP applications while keeping the
 * implementation lightweight and dependency-free.
 */

import { Kernel, checkBlocks } from "./base.js";

const SUPPORTED_NU = [0.5, 1.5, 2.5, Infinity];

export class Matern extends Kernel {
  /**
   * @param {number|number[]|Object} [lengthScaleOrOpts=1] - a length scale, one
   *   per input dimension (ARD), or an options object
   * @param {number|number[]} [lengthScaleOrOpts.lengthScale=1]
   * @param {number} [lengthScaleOrOpts.nu=1.5] - 0.5, 1.5, 2.5 or Infinity
   * @param {number} [lengthScaleOrOpts.variance=1] - also `amplitude`
   * @param {number[]} [lengthScaleOrOpts.blocks] - ARD by block: `blocks[i]` is
   *   the index into `lengthScale` that input dimension i uses, so a group of
   *   features shares one length scale. Three blocks cost three hyperparameters
   *   where per-feature ARD on 37 features costs 37, which is what keeps
   *   marginal-likelihood tuning honest on a few hundred rows.
   * @param {number[]} [lengthScaleOrOpts.lengthScaleBounds] - `[low, high]`
   *   honoured by hyperparameter optimization
   * @param {number[]} [lengthScaleOrOpts.varianceBounds] - likewise
   * @param {number} [nu=1.5]
   * @param {number} [variance=1]
   */
  constructor(lengthScaleOrOpts = 1.0, nu = 1.5, variance = 1.0) {
    super();
    // An ARD length scale is an ARRAY, and `typeof [] === 'object'`, so the
    // options branch would swallow it: `new Matern([1.1, 0.9], 2.5)` used to
    // find no .lengthScale on the array and silently fall back to the
    // defaults, discarding every argument. Arrays take the positional path.
    if (typeof lengthScaleOrOpts === "object" && !Array.isArray(lengthScaleOrOpts)) {
      const {
        lengthScale = 1.0,
        nu: nuOpt = 1.5,
        variance: varianceOpt = 1.0,
        amplitude,
        blocks,
        lengthScaleBounds,
        varianceBounds,
      } = lengthScaleOrOpts;
      this.lengthScale = lengthScale;
      this.nu = nuOpt;
      this.variance = amplitude ?? varianceOpt;
      // ARD by block: `blocks[i]` is the index into `lengthScale` that input
      // dimension i uses, so several dimensions share one length scale. A soil
      // block, a tissue block and a season, say, cost three hyperparameters
      // instead of one per feature, which is what keeps marginal-likelihood
      // tuning honest on a few hundred rows.
      this.blocks = blocks;
      // Optional [low, high] bounds honoured by hyperparameter optimization.
      this.lengthScaleBounds = lengthScaleBounds;
      this.varianceBounds = varianceBounds;
    } else {
      this.lengthScale = lengthScaleOrOpts;
      this.nu = nu;
      this.variance = variance;
    }
    checkBlocks(this, "Matern");

    if (!SUPPORTED_NU.includes(this.nu)) {
      throw new Error(
        `Unsupported Matérn ν=${this.nu}. Supported values: ${SUPPORTED_NU.join(", ")}`
      );
    }
  }

  compute(x1, x2) {
    // Length scale may be a scalar (isotropic) or a per-dimension array (ARD,
    // Automatic Relevance Determination). With ARD each input dimension gets
    // its own length scale; large values down-weight irrelevant features.
    const l = this.lengthScale;
    const isArr = Array.isArray(l);
    const blocks = this.blocks;

    // Distance with the length scale(s) folded in: Σ ((x1_i - x2_i) / l_i)².
    // With `blocks`, dimension i uses the length scale of its block.
    let scaledSq = 0;
    for (let i = 0; i < x1.length; i++) {
      const li = isArr ? l[blocks ? blocks[i] : i] : l;
      const s = (x1[i] - x2[i]) / li;
      scaledSq += s * s;
    }

    if (scaledSq === 0) {
      return this.variance;
    }

    if (this.nu === Infinity) {
      return this.variance * Math.exp(-scaledSq / 2);
    }

    const scale = Math.sqrt(2 * this.nu) * Math.sqrt(scaledSq);

    switch (this.nu) {
      case 0.5:
        return this.variance * Math.exp(-scale);
      case 1.5:
        return this.variance * (1 + scale) * Math.exp(-scale);
      case 2.5:
        return this.variance * (1 + scale + (scale * scale) / 3) * Math.exp(-scale);
      default:
        throw new Error("Unsupported ν for Matérn kernel");
    }
  }

  getParams() {
    const p = { lengthScale: this.lengthScale, nu: this.nu, variance: this.variance };
    if (this.blocks) p.blocks = this.blocks;
    if (this.lengthScaleBounds) p.lengthScaleBounds = this.lengthScaleBounds;
    if (this.varianceBounds) p.varianceBounds = this.varianceBounds;
    return p;
  }

  setParams({ lengthScale, nu, variance, amplitude }) {
    if (lengthScale !== undefined) this.lengthScale = lengthScale;
    if (nu !== undefined) {
      if (!SUPPORTED_NU.includes(nu)) {
        throw new Error(
          `Unsupported Matérn ν=${nu}. Supported values: ${SUPPORTED_NU.join(", ")}`
        );
      }
      this.nu = nu;
    }
    if (variance !== undefined) this.variance = variance;
    if (amplitude !== undefined) this.variance = amplitude;
  }
}
