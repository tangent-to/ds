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

// Most operations use built-in Math functions; the CoDA diagnostics
// (compositionalOutliers) additionally use a pseudo-inverse and the chi-squared
// distribution.
import { pseudoInverse } from "../core/linalg.js";
import { chisq } from "../stats/distribution.js";

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
 * Impute missing values in compositional data, respecting the simplex.
 *
 * Missing cells (`null`, `undefined` or `NaN`) are filled by an EM-style
 * iteration in centred-log-ratio (CLR) space: each incomplete row is updated so
 * that its CLR coordinates on the missing parts match the compositional
 * (CLR) mean of the complete observations, while the observed parts are
 * preserved. This is the log-ratio analogue of mean imputation and keeps the
 * imputed values strictly positive and coherent with the observed
 * sub-composition (cf. Martín-Fernández et al.; Palarea-Albaladejo &
 * Martín-Fernández, 2008). Combine with {@link multiplicativeReplacement} to
 * additionally handle essential zeros.
 *
 * @param {Array<Array<number>>} mat - Composition with missing entries.
 * @param {Object} [opts]
 * @param {number} [opts.maxIter=100] - Maximum EM iterations.
 * @param {number} [opts.tol=1e-9] - Convergence tolerance on the CLR mean.
 * @returns {Array<Array<number>>} Completed, strictly-positive composition.
 */
export function imputeMissing(mat, { maxIter = 100, tol = 1e-9 } = {}) {
  const { mat: M, was1d } = _ensureMatrix(mat);
  const n = M.length;
  const D = M[0].length;
  // Only genuinely missing cells (null / NaN) are imputed. Observed zeros are
  // left in place for multiplicativeReplacement, but they are excluded from the
  // (geometric) means below so they never poison a log — imputeMissing stays
  // numerically robust even when the data still contains essential zeros.
  const miss = M.map((row) => row.map((v) => v == null || Number.isNaN(v)));
  const anyMiss = miss.map((r) => r.some(Boolean));
  if (!anyMiss.some(Boolean)) return _restoreShape(M.map((r) => r.slice()), was1d);

  const X = M.map((r) => r.slice());
  const posLog = (i, j) => Number.isFinite(X[i][j]) && X[i][j] > 0; // usable in a log
  // Geometric-mean log of a row over its strictly-positive entries.
  const rowGLog = (i) => {
    let s = 0, c = 0;
    for (let j = 0; j < D; j++) if (posLog(i, j)) { s += Math.log(X[i][j]); c++; }
    return c ? s / c : 0;
  };

  // Initial fill: per-column geometric mean of the observed positive values.
  const colGM = new Array(D);
  for (let j = 0; j < D; j++) {
    let s = 0, c = 0;
    for (let i = 0; i < n; i++) if (!miss[i][j] && Number.isFinite(M[i][j]) && M[i][j] > 0) { s += Math.log(M[i][j]); c++; }
    colGM[j] = c ? Math.exp(s / c) : 1e-6;
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) if (miss[i][j]) X[i][j] = colGM[j];

  // Rows usable for the target CLR mean: no missing cells and all positive.
  const usable = [];
  for (let i = 0; i < n; i++) if (!anyMiss[i] && M[i].every((v) => Number.isFinite(v) && v > 0)) usable.push(i);

  let prevMean = null;
  for (let iter = 0; iter < maxIter; iter++) {
    const ref = usable.length >= 2 ? usable : X.map((_, i) => i);
    const meanClr = new Array(D).fill(0);
    for (const i of ref) {
      const gLog = rowGLog(i);
      for (let j = 0; j < D; j++) meanClr[j] += (posLog(i, j) ? Math.log(X[i][j]) : gLog) - gLog;
    }
    for (let j = 0; j < D; j++) meanClr[j] /= ref.length;

    // Update each incomplete row's missing parts to match the target CLR mean.
    for (let i = 0; i < n; i++) {
      if (!anyMiss[i]) continue;
      for (let inner = 0; inner < 100; inner++) {
        const gLog = rowGLog(i);
        let delta = 0;
        for (let j = 0; j < D; j++) {
          if (!miss[i][j]) continue;
          const nv = Math.exp(meanClr[j] + gLog);
          delta = Math.max(delta, Math.abs(Math.log(nv) - Math.log(X[i][j])));
          X[i][j] = nv;
        }
        if (delta < 1e-13) break;
      }
    }

    if (prevMean) {
      let md = 0;
      for (let j = 0; j < D; j++) md = Math.max(md, Math.abs(meanClr[j] - prevMean[j]));
      if (md < tol) break;
    }
    prevMean = meanClr;
  }
  return _restoreShape(X, was1d);
}

/**
 * Detect compositional outliers via the Mahalanobis distance in log-ratio
 * space, tested as a chi-squared variable (Filzmoser & Hron; Parent & Dafir,
 * 1992).
 *
 * Each observation's CLR (or ILR) vector is compared to a centroid using the
 * (pseudo-inverse) covariance; the squared Mahalanobis distance follows a
 * chi-squared distribution with `D − 1` degrees of freedom under compositional
 * normality. The centroid and covariance may be estimated from a reference
 * subpopulation (e.g. a high-yielding group) via `reference`.
 *
 * @param {Array<Array<number>>} mat - Strictly-positive composition.
 * @param {Object} [opts]
 * @param {boolean[]} [opts.reference=null] - Boolean mask selecting the rows
 *   that define the centroid/covariance (default: all rows).
 * @param {number} [opts.alpha=0.05] - Significance level for the outlier flag.
 * @param {"clr"|"ilr"} [opts.transform="clr"] - Log-ratio coordinates to use.
 * @returns {{ distances:number[], pValues:number[], outliers:boolean[],
 *   centroid:number[], covInverse:number[][], df:number }}
 */
export class CompositionalOutlierDetector {
  /**
   * @param {Object} [opts]
   * @param {"clr"|"ilr"} [opts.transform="clr"] - Log-ratio coordinates to use.
   * @param {number} [opts.alpha=0.05] - Significance level for the outlier flag.
   */
  constructor({ transform = "clr", alpha = 0.05 } = {}) {
    this.transform = transform;
    this.alpha = alpha;
    this.fitted = false;
  }

  /**
   * Estimate the centroid and (pseudo-inverse) covariance in log-ratio space
   * from a reference composition — e.g. a healthy / high-yielding subpopulation.
   * @param {Array<Array<number>>} mat - Strictly-positive reference composition.
   * @returns {CompositionalOutlierDetector} this
   */
  fit(mat) {
    const { mat: M } = _ensureMatrix(mat);
    if (M.length < 2) throw new Error("CompositionalOutlierDetector.fit: need at least 2 reference rows");
    const Y = this.transform === "ilr" ? ilr(M) : clr(M);
    const d = Y[0].length;
    this.nParts = M[0].length;
    this.dim = d;
    this.df = this.transform === "ilr" ? d : this.nParts - 1;

    const centroid = new Array(d).fill(0);
    for (const y of Y) for (let j = 0; j < d; j++) centroid[j] += y[j];
    for (let j = 0; j < d; j++) centroid[j] /= Y.length;
    this.center = centroid;

    const cov = Array.from({ length: d }, () => new Array(d).fill(0));
    for (const y of Y) {
      const dv = y.map((v, j) => v - centroid[j]);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a][b] += dv[a] * dv[b];
    }
    const denom = Math.max(1, Y.length - 1);
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a][b] /= denom;
    // CLR covariance is singular (rank D−1); use the Moore-Penrose pseudo-inverse.
    const piv = pseudoInverse(cov);
    this.covInverse = typeof piv.to2DArray === "function" ? piv.to2DArray() : piv;
    this.fitted = true;
    return this;
  }

  /** Squared Mahalanobis distance in log-ratio space for each row of `mat`. */
  distance(mat) {
    if (!this.fitted) throw new Error("CompositionalOutlierDetector: call fit() first");
    const { mat: M } = _ensureMatrix(mat);
    const Y = this.transform === "ilr" ? ilr(M) : clr(M);
    return Y.map((y) => {
      const dv = y.map((v, j) => v - this.center[j]);
      let s = 0;
      for (let a = 0; a < this.dim; a++) {
        let row = 0;
        for (let b = 0; b < this.dim; b++) row += this.covInverse[a][b] * dv[b];
        s += dv[a] * row;
      }
      return s;
    });
  }

  /** Chi-squared p-value (1 − CDF) for each row's Mahalanobis distance. */
  pValue(mat) {
    return this.distance(mat).map((D2) => 1 - chisq.cdf(D2, { df: this.df }));
  }

  /**
   * Test rows for compositional outlyingness against the fitted reference.
   * @param {Array<Array<number>>} mat - Composition(s) to test (e.g. all
   *   experimental samples, or external standards to project).
   * @returns {{ distances:number[], pValues:number[], outliers:boolean[], df:number }}
   */
  test(mat) {
    const distances = this.distance(mat);
    const pValues = distances.map((D2) => 1 - chisq.cdf(D2, { df: this.df }));
    return { distances, pValues, outliers: pValues.map((p) => p < this.alpha), df: this.df };
  }
}

/**
 * Detect compositional outliers via the Mahalanobis distance in log-ratio
 * space, tested as a chi-squared variable (Filzmoser & Hron; Parent & Dafir,
 * 1992). Convenience wrapper around {@link CompositionalOutlierDetector} that
 * fits on `mat` (or a `reference` subset of it) and tests `mat`.
 *
 * For testing *new* points against the fitted reference (e.g. external
 * standards), fit a detector once and call `.test(newComposition)` — no manual
 * projection needed.
 *
 * @param {Array<Array<number>>} mat - Strictly-positive composition.
 * @param {Object} [opts]
 * @param {boolean[]} [opts.reference=null] - Mask selecting the rows that define
 *   the centroid/covariance (default: all rows).
 * @param {number} [opts.alpha=0.05]
 * @param {"clr"|"ilr"} [opts.transform="clr"]
 * @returns {{ distances:number[], pValues:number[], outliers:boolean[],
 *   df:number, center:number[], covInverse:number[][],
 *   detector:CompositionalOutlierDetector }}
 */
export function compositionalOutliers(mat, { reference = null, alpha = 0.05, transform = "clr" } = {}) {
  const { mat: M } = _ensureMatrix(mat);
  const refMat = reference ? M.filter((_, i) => reference[i]) : M;
  const detector = new CompositionalOutlierDetector({ transform, alpha }).fit(refMat);
  return { ...detector.test(M), center: detector.center, covInverse: detector.covInverse, detector };
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
  imputeMissing,
  compositionalOutliers,
  CompositionalOutlierDetector,
};
