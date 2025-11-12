/**
 * Compositional Data Analysis Utilities
 *
 * NumPy-based implementations for compositional data transformations,
 * adapted from scikit-bio under the Modified BSD License.
 * Translated to JavaScript for use in the tangent-ds library.
 *
 * This module provides transformations for compositional data analysis,
 * including log-ratio transformations (CLR, ALR, ILR) and related utilities.
 */

// All operations use built-in Math functions, no linalg imports needed

/**
 * Validates that all values in the array are positive
 * @param {Array|Array<Array>} mat - Input array
 * @param {string} name - Name for error messages
 * @throws {Error} If any values are negative
 */
function _checkPositive(mat, name = 'mat') {
  const arr = Array.isArray(mat[0]) ? mat.flat() : mat;
  if (arr.some((x) => x < 0)) {
    throw new Error(`${name} cannot contain negative values`);
  }
}

/**
 * Computes geometric mean of rows
 * @param {Array<Array>} mat - 2D array
 * @returns {Array} Geometric means of each row
 */
function _gmean(mat) {
  return mat.map((row) => {
    const logSum = row.reduce((sum, x) => sum + Math.log(x), 0);
    return Math.exp(logSum / row.length);
  });
}

/**
 * Generates default orthonormal basis using Gram-Schmidt process
 * @param {number} D - Dimension
 * @returns {Array<Array>} Orthonormal basis matrix (D-1) x D
 */
function _defaultBasis(D) {
  const basis = [];
  for (let i = 0; i < D - 1; i++) {
    const row = new Array(D).fill(0);
    for (let j = 0; j <= i; j++) {
      row[j] = 1 / Math.sqrt(i + 1) / Math.sqrt(i + 2);
    }
    row[i + 1] = -(i + 1) / Math.sqrt(i + 1) / Math.sqrt(i + 2);
    basis.push(row);
  }
  return basis;
}

/**
 * Ensures input is 2D array and tracks original shape
 * @param {Array|Array<Array>} X - Input array
 * @returns {Object} {mat: 2D array, was1d: boolean}
 */
function _ensureMatrix(X) {
  if (!Array.isArray(X[0])) {
    // 1D array - convert to 2D
    return { mat: [X], was1d: true };
  }
  return { mat: X, was1d: false };
}

/**
 * Restores original dimensionality
 * @param {Array<Array>} mat - 2D array
 * @param {boolean} was1d - Whether original was 1D
 * @returns {Array|Array<Array>} Array in original dimensionality
 */
function _restoreShape(mat, was1d) {
  return was1d ? mat[0] : mat;
}

/**
 * Normalizes rows to sum to 1 (closure operation)
 * @param {Array|Array<Array>} mat - Input compositional data
 * @returns {Array|Array<Array>} Closed composition (rows sum to 1)
 */
export function closure(mat) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);
  _checkPositive(mat2d, 'mat');

  const closed = mat2d.map((row) => {
    const rowSum = row.reduce((sum, x) => sum + x, 0);
    if (rowSum === 0) {
      throw new Error('Cannot close composition with zero sum');
    }
    return row.map((x) => x / rowSum);
  });

  return _restoreShape(closed, was1d);
}

/**
 * Replaces zeros with small delta values before closure
 * @param {Array|Array<Array>} mat - Input compositional data
 * @param {number} delta - Replacement value for zeros (default: 1e-6)
 * @returns {Array|Array<Array>} Composition with zeros replaced
 */
export function multiplicativeReplacement(mat, delta = 1e-6) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);
  _checkPositive(mat2d, 'mat');

  const replaced = mat2d.map((row) => {
    const numZeros = row.filter((x) => x === 0).length;
    if (numZeros === 0) {
      return row;
    }

    const numNonZeros = row.length - numZeros;
    const adjustment = (delta * numZeros) / numNonZeros;

    return row.map((x) => x === 0 ? delta : x - adjustment);
  });

  return _restoreShape(closure(replaced), was1d);
}

/**
 * Raises components to a power and renormalizes
 * @param {Array|Array<Array>} mat - Input compositional data
 * @param {number} power - Exponent
 * @returns {Array|Array<Array>} Powered and renormalized composition
 */
export function power(mat, pow) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);
  _checkPositive(mat2d, 'mat');

  const powered = mat2d.map((row) => row.map((x) => Math.pow(x, pow)));

  return _restoreShape(closure(powered), was1d);
}

/**
 * Centers compositions by geometric mean
 * @param {Array|Array<Array>} mat - Input compositional data
 * @returns {Array|Array<Array>} Centered composition
 */
export function center(mat) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);
  _checkPositive(mat2d, 'mat');

  // Compute geometric mean across all rows
  const allValues = mat2d.flat();
  const overallGmean = Math.exp(
    allValues.reduce((sum, x) => sum + Math.log(x), 0) / allValues.length,
  );

  const centered = mat2d.map((row) => row.map((x) => x / overallGmean));

  return _restoreShape(closure(centered), was1d);
}

/**
 * Alias for center
 */
export const centralize = center;

/**
 * Centered log-ratio transformation (CLR)
 * @param {Array|Array<Array>} mat - Input compositional data (positive values)
 * @param {boolean} handleZeros - If true, replace zeros before transform
 * @param {number} delta - Replacement value for zeros
 * @returns {Array|Array<Array>} CLR-transformed data
 */
export function clr(mat, handleZeros = false, delta = 1e-6) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  let processed = mat2d;
  if (handleZeros) {
    processed = multiplicativeReplacement(processed, delta);
    processed = _ensureMatrix(processed).mat;
  } else {
    _checkPositive(processed, 'mat');
  }

  const transformed = processed.map((row) => {
    const gmean = Math.exp(row.reduce((sum, x) => sum + Math.log(x), 0) / row.length);
    return row.map((x) => Math.log(x / gmean));
  });

  return _restoreShape(transformed, was1d);
}

/**
 * Inverse centered log-ratio transformation
 * @param {Array|Array<Array>} mat - CLR-transformed data
 * @returns {Array|Array<Array>} Composition (rows sum to 1)
 */
export function clrInv(mat) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  const transformed = mat2d.map((row) => row.map((x) => Math.exp(x)));

  return _restoreShape(closure(transformed), was1d);
}

/**
 * Additive log-ratio transformation (ALR)
 * Uses the last component as the reference denominator
 * @param {Array|Array<Array>} mat - Input compositional data
 * @param {number|null} denomIdx - Index of denominator component (default: last)
 * @param {boolean} handleZeros - If true, replace zeros before transform
 * @param {number} delta - Replacement value for zeros
 * @returns {Array|Array<Array>} ALR-transformed data (dimension reduced by 1)
 */
export function alr(mat, denomIdx = null, handleZeros = false, delta = 1e-6) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  let processed = mat2d;
  if (handleZeros) {
    processed = multiplicativeReplacement(processed, delta);
    processed = _ensureMatrix(processed).mat;
  } else {
    _checkPositive(processed, 'mat');
  }

  const D = processed[0].length;
  const denom = denomIdx === null ? D - 1 : denomIdx;

  if (denom < 0 || denom >= D) {
    throw new Error(`denomIdx ${denom} is out of bounds for dimension ${D}`);
  }

  const transformed = processed.map((row) => {
    const result = [];
    for (let i = 0; i < D; i++) {
      if (i !== denom) {
        result.push(Math.log(row[i] / row[denom]));
      }
    }
    return result;
  });

  return _restoreShape(transformed, was1d);
}

/**
 * Inverse additive log-ratio transformation
 * @param {Array|Array<Array>} mat - ALR-transformed data
 * @param {number|null} denomIdx - Index where denominator was (default: last)
 * @returns {Array|Array<Array>} Composition (rows sum to 1)
 */
export function alrInv(mat, denomIdx = null) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  const Dm1 = mat2d[0].length;
  const D = Dm1 + 1;
  const denom = denomIdx === null ? D - 1 : denomIdx;

  const transformed = mat2d.map((row) => {
    const result = new Array(D);
    let j = 0;
    for (let i = 0; i < D; i++) {
      if (i === denom) {
        result[i] = 1; // Reference component
      } else {
        result[i] = Math.exp(row[j]);
        j++;
      }
    }
    return result;
  });

  return _restoreShape(closure(transformed), was1d);
}

/**
 * Constructs orthonormal basis from sequential binary partition
 * @param {Array<Array>} partition - Sequential binary partition matrix
 * @returns {Array<Array>} Orthonormal basis for ILR
 */
export function sbpBasis(partition) {
  const numParts = partition.length;
  const D = partition[0].length;
  const basis = [];

  for (let i = 0; i < numParts; i++) {
    const row = partition[i];
    const nPlus = row.filter((x) => x === 1).length;
    const nMinus = row.filter((x) => x === -1).length;

    if (nPlus === 0 || nMinus === 0) {
      throw new Error(`Partition row ${i} must have both +1 and -1 values`);
    }

    const coefficient = Math.sqrt((nPlus * nMinus) / (nPlus + nMinus));
    const basisRow = row.map((x) => {
      if (x === 1) return coefficient / nPlus;
      if (x === -1) return -coefficient / nMinus;
      return 0;
    });

    basis.push(basisRow);
  }

  return basis;
}

/**
 * Isometric log-ratio transformation (ILR)
 * @param {Array|Array<Array>} mat - Input compositional data
 * @param {Array<Array>|null} basis - Orthonormal basis (default: Gram-Schmidt basis)
 * @param {boolean} handleZeros - If true, replace zeros before transform
 * @param {number} delta - Replacement value for zeros
 * @returns {Array|Array<Array>} ILR-transformed data
 */
export function ilr(mat, basis = null, handleZeros = false, delta = 1e-6) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  let processed = mat2d;
  if (handleZeros) {
    processed = multiplicativeReplacement(processed, delta);
    processed = _ensureMatrix(processed).mat;
  } else {
    _checkPositive(processed, 'mat');
  }

  const D = processed[0].length;
  const psi = basis === null ? _defaultBasis(D) : basis;

  if (psi.length !== D - 1 || psi[0].length !== D) {
    throw new Error(`Basis must be (D-1) x D matrix, got ${psi.length} x ${psi[0].length}`);
  }

  // Apply CLR first
  const clrMat = clr(processed);
  const clrMat2d = _ensureMatrix(clrMat).mat;

  // Multiply CLR-transformed data by basis transpose
  const transformed = clrMat2d.map((row) => {
    return psi.map((basisRow) => {
      return row.reduce((sum, val, idx) => sum + val * basisRow[idx], 0);
    });
  });

  return _restoreShape(transformed, was1d);
}

/**
 * Inverse isometric log-ratio transformation
 * @param {Array|Array<Array>} mat - ILR-transformed data
 * @param {Array<Array>|null} basis - Orthonormal basis used in forward transform
 * @returns {Array|Array<Array>} Composition (rows sum to 1)
 */
export function ilrInv(mat, basis = null) {
  const { mat: mat2d, was1d } = _ensureMatrix(mat);

  const Dm1 = mat2d[0].length;
  const D = Dm1 + 1;
  const psi = basis === null ? _defaultBasis(D) : basis;

  if (psi.length !== D - 1 || psi[0].length !== D) {
    throw new Error(`Basis must be (D-1) x D matrix, got ${psi.length} x ${psi[0].length}`);
  }

  // Multiply ILR coordinates by basis (mat2d @ psi)
  const clrMat = mat2d.map((row) => {
    const result = new Array(D).fill(0);
    for (let j = 0; j < D; j++) {
      for (let i = 0; i < Dm1; i++) {
        result[j] += row[i] * psi[i][j];
      }
    }
    return result;
  });

  // Apply inverse CLR
  return _restoreShape(clrInv(clrMat), was1d);
}

/**
 * Computes inner product in the Aitchison simplex
 * @param {Array|Array<Array>} x - First composition
 * @param {Array|Array<Array>} y - Second composition
 * @returns {number|Array} Inner product(s)
 */
export function inner(x, y) {
  const { mat: xMat, was1d: x1d } = _ensureMatrix(x);
  const { mat: yMat, was1d: y1d } = _ensureMatrix(y);

  _checkPositive(xMat, 'x');
  _checkPositive(yMat, 'y');

  if (xMat.length !== yMat.length || xMat[0].length !== yMat[0].length) {
    throw new Error('x and y must have the same dimensions');
  }

  const D = xMat[0].length;

  const products = xMat.map((xRow, i) => {
    const yRow = yMat[i];
    const xGmean = Math.exp(xRow.reduce((sum, val) => sum + Math.log(val), 0) / D);
    const yGmean = Math.exp(yRow.reduce((sum, val) => sum + Math.log(val), 0) / D);

    let sum = 0;
    for (let j = 0; j < D; j++) {
      sum += Math.log(xRow[j] / xGmean) * Math.log(yRow[j] / yGmean);
    }
    return sum;
  });

  return (x1d && y1d) ? products[0] : products;
}

/**
 * Export all functions
 */
export default {
  closure,
  multiplicativeReplacement,
  power,
  center,
  centralize,
  clr,
  clrInv,
  alr,
  alrInv,
  ilr,
  ilrInv,
  sbpBasis,
  inner,
};
