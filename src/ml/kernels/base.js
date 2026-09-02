/**
 * Base Kernel class for Gaussian Processes
 * 
 * Kernels define the covariance function between input points.
 * They measure similarity: points close together have high covariance.
 */

import { toMatrix } from '../../core/linalg.js';

/**
 * Abstract base class for GP kernels
 */
export class Kernel {
  /**
   * Compute covariance between two points
   * @param {Array<number>} x1 - First point
   * @param {Array<number>} x2 - Second point
   * @returns {number} Covariance value
   */
  compute(_x1, _x2) {
    throw new Error('Kernel.compute() must be implemented by subclass');
  }

  /**
   * Compute covariance matrix between sets of points
   * @param {Array<Array<number>>|Matrix} X1 - First set of points (n1 x d)
   * @param {Array<Array<number>>|Matrix} [X2] - Second set of points (n2 x d). If omitted, computes K(X1, X1)
   * @returns {Matrix} Covariance matrix (n1 x n2)
   */
  call(X1, X2 = null) {
    const M1 = toMatrix(X1);
    const M2 = X2 ? toMatrix(X2) : M1;
    const symmetric = X2 === null;

    const n1 = M1.rows;
    const n2 = M2.rows;

    // Extract rows once. getRow() allocates a fresh array, and doing it
    // inside the inner loop allocated O(n1·n2) throwaway rows per kernel
    // build — a large GC load given call() runs on every fit and predict.
    const rows2 = new Array(n2);
    for (let j = 0; j < n2; j++) rows2[j] = M2.getRow(j);
    let rows1 = rows2;
    if (!symmetric) {
      rows1 = new Array(n1);
      for (let i = 0; i < n1; i++) rows1[i] = M1.getRow(i);
    }

    const K = new Array(n1);
    for (let i = 0; i < n1; i++) {
      K[i] = new Array(n2);
      const row1 = rows1[i];

      for (let j = 0; j < n2; j++) {
        // Use symmetry if computing K(X, X)
        if (symmetric && j < i) {
          K[i][j] = K[j][i];
        } else {
          K[i][j] = this.compute(row1, rows2[j]);
        }
      }
    }

    return toMatrix(K);
  }

  /**
   * Get kernel hyperparameters
   * @returns {Object} Hyperparameters
   */
  getParams() {
    return {};
  }

  /**
   * Set kernel hyperparameters
   * @param {Object} params - New parameters
   */
  setParams(_params) {
    // Override in subclasses
  }

  /**
   * Clone the kernel with the same parameters
   * @returns {Kernel} New kernel instance
   */
  clone() {
    return new this.constructor(this.getParams());
  }
}


/**
 * Validate a `blocks` map on an ARD kernel: one entry per input dimension,
 * each an index into `lengthScale`, every length scale used. Shared by RBF
 * and Matérn.
 * @param {Object} k - the kernel
 * @param {string} name - for messages
 */
export function checkBlocks(k, name) {
  if (k.blocks === undefined) return;
  const l = k.lengthScale;
  if (!Array.isArray(l)) {
    throw new Error(`${name}: blocks needs an array lengthScale with one entry per block`);
  }
  if (!Array.isArray(k.blocks) || k.blocks.some((b) => !Number.isInteger(b) || b < 0 || b >= l.length)) {
    throw new Error(`${name}: blocks must be an array of indices into lengthScale (0 to ${l.length - 1})`);
  }
  const used = new Set(k.blocks);
  if (used.size !== l.length) {
    throw new Error(`${name}: lengthScale has ${l.length} entries but blocks uses ${used.size} of them`);
  }
}
