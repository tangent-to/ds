/**
 * Linear algebra operations wrapping @tangent.to/lina
 * This abstraction allows future backend swaps (e.g., WASM)
 */

import {
  cholesky as linaCholesky,
  choleskySolve as linaCholeskySolve,
  eigSym as linaEigSym,
  eigSymGeneralized as linaEigSymGeneralized,
  invSqrtSym as linaInvSqrtSym,
  inv as linaInv,
  pinv as linaPinv,
  solve as linaSolve,
  svd as linaSvd,
} from '@tangent.to/lina';
import { asArray, Matrix } from './matrix.js';

/**
 * Convert array-like structure to Matrix
 * @param {Array<Array<number>>|Matrix} data - Input data
 * @returns {Matrix} Matrix object
 */
export function toMatrix(data) {
  if (data instanceof Matrix) {
    return data;
  }
  return new Matrix(data);
}

/**
 * Solve least squares problem: minimize ||Ax - b||^2
 * @param {Array<Array<number>>|Matrix} A - Design matrix
 * @param {Array<number>|Array<Array<number>>|Matrix} b - Target vector/matrix
 * @returns {Matrix} Solution x
 */
export function solveLeastSquares(A, b) {
  const matA = toMatrix(A);
  const matB = Array.isArray(b) && !Array.isArray(b[0])
    ? Matrix.columnVector(b)
    : toMatrix(b);

  // Solve using normal equations: (A'A)x = A'b
  const At = matA.transpose();
  const AtA = At.mmul(matA);
  const Atb = At.mmul(matB);

  try {
    return new Matrix(linaSolve(AtA.data, Atb.data));
  } catch (_e) {
    // If singular, use minimum-norm solution via pseudoinverse
    return new Matrix(linaPinv(matA.data)).mmul(matB);
  }
}

/**
 * Compute covariance matrix
 * @param {Array<Array<number>>|Matrix} data - Data matrix (rows = observations)
 * @param {boolean} center - If true, center the data
 * @returns {Matrix} Covariance matrix
 */
export function covarianceMatrix(data, center = true) {
  let mat = toMatrix(data);

  if (center) {
    // Center each column
    const means = mat.mean('column');
    mat = mat.clone();
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.columns; j++) {
        mat.set(i, j, mat.get(i, j) - means[j]);
      }
    }
  }

  const n = mat.rows;
  return mat.transpose().mmul(mat).div(n - 1);
}

/**
 * Singular Value Decomposition (thin: U is m×k, V is n×k, k = min(m, n))
 * @param {Array<Array<number>>|Matrix} data - Input matrix
 * @returns {Object} {U, s, V} where data ≈ U * diag(s) * V'
 */
export function svd(data) {
  const { U, s, V } = linaSvd(asArray(toMatrix(data)));
  return {
    U: new Matrix(U),
    s,
    V: new Matrix(V),
  };
}

/**
 * Eigenvalue decomposition of a symmetric matrix.
 * Eigenvalues are returned in descending order; eigenvectors are the
 * columns of `vectors`. Throws for non-symmetric input.
 * @param {Array<Array<number>>|Matrix} data - Symmetric square matrix
 * @returns {Object} {values, vectors}
 */
export function eig(data) {
  const { values, vectors } = linaEigSym(asArray(toMatrix(data)));
  return {
    values,
    vectors: new Matrix(vectors),
  };
}

/**
 * Generalized symmetric eigendecomposition: solve A x = lambda B x for
 * symmetric A and symmetric positive (semi)definite B. Eigenvalues are
 * returned in descending order; eigenvectors are the columns of `vectors`.
 *
 * When B is positive definite the vectors are B-orthonormal (x'Bx = 1), as
 * from scipy's eigh(A, B). When B is singular the problem is solved on
 * range(B) and the vectors have unit euclidean length instead; `definite`
 * reports which case applied.
 *
 * @param {Array<Array<number>>|Matrix} A - Symmetric matrix
 * @param {Array<Array<number>>|Matrix} B - Symmetric positive (semi)definite matrix
 * @returns {Object} {values, vectors, definite}
 */
export function eigGeneralized(A, B) {
  const { values, vectors, definite } = linaEigSymGeneralized(
    asArray(toMatrix(A)),
    asArray(toMatrix(B)),
  );
  return { values, vectors: new Matrix(vectors), definite };
}

/**
 * Inverse square root of a symmetric positive semidefinite matrix: the
 * symmetric W with W A W = I on A's range, and 0 on its null space
 * @param {Array<Array<number>>|Matrix} data - Symmetric positive semidefinite matrix
 * @returns {Matrix} Symmetric inverse square root
 */
export function symmetricInverseSqrt(data) {
  return new Matrix(linaInvSqrtSym(asArray(toMatrix(data))));
}

/**
 * Matrix multiplication
 * @param {Array<Array<number>>|Matrix} A - First matrix
 * @param {Array<Array<number>>|Matrix} B - Second matrix
 * @returns {Matrix} A * B
 */
export function mmul(A, B) {
  return toMatrix(A).mmul(toMatrix(B));
}

/**
 * Matrix transpose
 * @param {Array<Array<number>>|Matrix} data - Input matrix
 * @returns {Matrix} Transposed matrix
 */
export function transpose(data) {
  return toMatrix(data).transpose();
}

/**
 * Matrix inverse
 * @param {Array<Array<number>>|Matrix} data - Square matrix
 * @returns {Matrix} Inverse matrix
 */
export function inverse(data) {
  return new Matrix(linaInv(asArray(toMatrix(data))));
}

/**
 * Solve the linear system Ax = b (square A); throws if singular
 * @param {Array<Array<number>>|Matrix} A - Square matrix
 * @param {Array<Array<number>>|Matrix} b - Right-hand side (column(s))
 * @returns {Matrix} Solution x
 */
export function solve(A, b) {
  return new Matrix(linaSolve(asArray(toMatrix(A)), asArray(toMatrix(b))));
}

/**
 * Cholesky factorization of a symmetric positive definite matrix
 * @param {Array<Array<number>>|Matrix} data - Symmetric positive definite matrix
 * @returns {Matrix} Lower triangular L with data = L * L'
 * @throws {Error} When the matrix is not symmetric or not positive definite
 */
export function cholesky(data) {
  return new Matrix(linaCholesky(asArray(toMatrix(data))));
}

/**
 * Solve A x = b from the Cholesky factor L of A, by forward then back
 * substitution
 * @param {Array<Array<number>>|Matrix} L - Lower triangular factor
 * @param {Array<number>} b - Right-hand side vector
 * @returns {Array<number>} Solution x
 */
export function choleskySolve(L, b) {
  return linaCholeskySolve(asArray(toMatrix(L)), b);
}

/**
 * Moore-Penrose pseudoinverse via SVD with a singular-value cutoff
 * scaled by the matrix size and largest singular value (numpy
 * convention), so near-zero singular values are zeroed instead of
 * inverted into garbage for nearly rank-deficient matrices.
 * @param {Array<Array<number>>|Matrix} data - Input matrix
 * @returns {Matrix} Pseudoinverse
 */
export function pseudoInverse(data) {
  return new Matrix(linaPinv(asArray(toMatrix(data))));
}

/**
 * SVD with the decomposition-object interface of ml-matrix, for
 * least-squares solves and pseudoinverses reusing one factorization.
 */
export class SingularValueDecomposition {
  /**
   * @param {Array<Array<number>>|Matrix} data - Input matrix (any shape)
   */
  constructor(data) {
    const { U, s, V } = linaSvd(asArray(toMatrix(data)));
    this._U = U;
    this._s = s;
    this._V = V;
    this._m = U.length;
    this._n = V.length;
  }

  get leftSingularVectors() {
    return new Matrix(this._U);
  }

  get rightSingularVectors() {
    return new Matrix(this._V);
  }

  get diagonal() {
    return this._s;
  }

  _cutoff() {
    return Math.max(this._m, this._n) * Number.EPSILON * (this._s[0] || 0);
  }

  /**
   * Minimum-norm least-squares solution of A x = b
   * @param {Array<Array<number>>|Matrix} b - Right-hand side (column(s))
   * @returns {Matrix} Solution x
   */
  solve(b) {
    const cutoff = this._cutoff();
    const sInv = this._s.map((v) => (v > cutoff ? 1 / v : 0));
    // x = V * diag(1/s) * U' * b
    const Utb = new Matrix(this._U).transpose().mmul(toMatrix(b));
    for (let i = 0; i < sInv.length; i++) {
      for (let j = 0; j < Utb.columns; j++) {
        Utb.set(i, j, Utb.get(i, j) * sInv[i]);
      }
    }
    return new Matrix(this._V).mmul(Utb);
  }

  /**
   * Pseudoinverse from the computed factorization
   * @returns {Matrix} Pseudoinverse
   */
  inverse() {
    const cutoff = this._cutoff();
    const sInv = this._s.map((v) => (v > cutoff ? 1 / v : 0));
    // pinv = V * diag(1/s) * U'
    const VS = new Matrix(this._V);
    for (let i = 0; i < VS.rows; i++) {
      for (let j = 0; j < sInv.length; j++) {
        VS.set(i, j, VS.get(i, j) * sInv[j]);
      }
    }
    return VS.mmul(new Matrix(this._U).transpose());
  }
}

export { Matrix };
