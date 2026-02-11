/**
 * Matérn Kernel
 *
 * Supports ν = 0.5, 1.5, 2.5, or Infinity (which reduces to the RBF kernel).
 * These common cases cover most GP applications while keeping the
 * implementation lightweight and dependency-free.
 */

import { Kernel } from "./base.js";

const SUPPORTED_NU = [0.5, 1.5, 2.5, Infinity];

export class Matern extends Kernel {
  constructor(lengthScaleOrOpts = 1.0, nu = 1.5, variance = 1.0) {
    super();
    if (typeof lengthScaleOrOpts === "object") {
      const {
        lengthScale = 1.0,
        nu: nuOpt = 1.5,
        variance: varianceOpt = 1.0,
        amplitude,
      } = lengthScaleOrOpts;
      this.lengthScale = lengthScale;
      this.nu = nuOpt;
      this.variance = amplitude ?? varianceOpt;
    } else {
      this.lengthScale = lengthScaleOrOpts;
      this.nu = nu;
      this.variance = variance;
    }

    if (!SUPPORTED_NU.includes(this.nu)) {
      throw new Error(
        `Unsupported Matérn ν=${this.nu}. Supported values: ${SUPPORTED_NU.join(", ")}`
      );
    }
  }

  compute(x1, x2) {
    let squaredDistance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = x1[i] - x2[i];
      squaredDistance += diff * diff;
    }

    const r = Math.sqrt(squaredDistance);
    if (r === 0) {
      return this.variance;
    }

    const scale = Math.sqrt(2 * (this.nu === Infinity ? 1 : this.nu)) * r / this.lengthScale;

    switch (this.nu) {
      case 0.5:
        return this.variance * Math.exp(-scale);
      case 1.5:
        return this.variance * (1 + scale) * Math.exp(-scale);
      case 2.5:
        return this.variance * (1 + scale + (scale * scale) / 3) * Math.exp(-scale);
      case Infinity:
        return this.variance * Math.exp(-squaredDistance / (2 * this.lengthScale * this.lengthScale));
      default:
        throw new Error("Unsupported ν for Matérn kernel");
    }
  }

  getParams() {
    return {
      lengthScale: this.lengthScale,
      nu: this.nu,
      variance: this.variance,
    };
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
