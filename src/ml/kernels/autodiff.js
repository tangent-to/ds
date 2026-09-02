/**
 * Kernel matrices as differentiable expressions.
 *
 * The GP hyperparameter optimizer needs ∂/∂θ of the log marginal likelihood.
 * Hand-deriving that is only tractable for a kernel or two — in this package it
 * was written for the Matérn family alone, and every other kernel fell back to
 * a derivative-free search that stops at a visibly worse optimum. Rebuilding
 * the kernel matrix in @tangent.to/grad ops gives every kernel an exact
 * gradient from the same code.
 *
 * The pairwise geometry — squared differences per dimension, total squared
 * distances, Euclidean distances, inner products — does NOT depend on the
 * hyperparameters, so it is computed once per fit by {@link kernelConstants}
 * and reused across every optimizer step. Only a handful of elementwise ops on
 * n×n constants are taped per evaluation.
 *
 * The forward math here mirrors each kernel's own `compute()`. That
 * duplication is the price of not rewriting every kernel class to be generic
 * over its arithmetic, and it is guarded by tests asserting this builder
 * reproduces `kernel.call(X)` exactly.
 */

import { add, div, exp, log, matmul, mul, reshape, slice, sqrt, square, sub } from '@tangent.to/grad';

/**
 * Precompute everything about the inputs that no hyperparameter can change.
 *
 * @param {Array<Array<number>>} X - training inputs (n × d)
 * @returns {{n: number, d: number, perDim: Array<Array<Array<number>>>,
 *   total: Array<Array<number>>, dist: Array<Array<number>>,
 *   dot: Array<Array<number>>}}
 */
export function kernelConstants(X) {
  const n = X.length;
  const d = X[0].length;
  const perDim = [];
  for (let k = 0; k < d; k++) {
    const M = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const delta = X[i][k] - X[j][k];
        M[i][j] = delta * delta;
      }
    }
    perDim.push(M);
  }
  const total = Array.from({ length: n }, () => new Array(n).fill(0));
  const dist = Array.from({ length: n }, () => new Array(n).fill(0));
  const dot = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      let p = 0;
      for (let k = 0; k < d; k++) {
        s += perDim[k][i][j];
        p += X[i][k] * X[j][k];
      }
      total[i][j] = s;
      dist[i][j] = Math.sqrt(s);
      dot[i][j] = p;
    }
  }
  return { n, d, perDim, total, dist, dot };
}

/**
 * Pull hyperparameter `i` out of the packed vector as a SCALAR Var.
 * `slice` yields a length-1 vector; reshaping to the empty shape makes it a
 * scalar, which is what the elementwise ops broadcast against an n×n matrix.
 * @private
 */
function at(theta, i) {
  return reshape(slice(theta, [i], [1]), []);
}

/**
 * Keeps sqrt() differentiable at a zero distance; see the Matérn branch below.
 * @private
 */
const MATERN_SQRT_FLOOR = 1e-30;

/**
 * Σ_k D2[k] / l_k², the length-scale-weighted squared distance shared by the
 * RBF and Matérn families. Handles both an isotropic scale and an ARD vector.
 * @private
 */
function scaledSquared(consts, theta, offset, kernel) {
  const isArd = Array.isArray(kernel.lengthScale);
  if (!isArd) {
    const l = at(theta, offset);
    return { value: div(consts.total, square(l)), next: offset + 1 };
  }
  if (kernel.blocks) {
    // One length scale per block: the squared distances of a block's
    // dimensions are summed once, as a constant, and divided by that block's
    // scale. The sums are cached on consts, keyed by the block map, since
    // this is rebuilt on every gradient evaluation.
    const key = kernel.blocks.join(',');
    consts.blockSums = consts.blockSums || {};
    if (!consts.blockSums[key]) {
      const nBlocks = kernel.lengthScale.length;
      const sums = Array.from({ length: nBlocks }, () =>
        Array.from({ length: consts.n }, () => new Array(consts.n).fill(0)));
      for (let k = 0; k < consts.d; k++) {
        const b = kernel.blocks[k];
        const M = consts.perDim[k];
        const S = sums[b];
        for (let i = 0; i < consts.n; i++) for (let j = 0; j < consts.n; j++) S[i][j] += M[i][j];
      }
      consts.blockSums[key] = sums;
    }
    const sums = consts.blockSums[key];
    let acc = null;
    for (let b = 0; b < sums.length; b++) {
      const term = div(sums[b], square(at(theta, offset + b)));
      acc = acc === null ? term : add(acc, term);
    }
    return { value: acc, next: offset + sums.length };
  }
  let acc = null;
  for (let k = 0; k < consts.d; k++) {
    const term = div(consts.perDim[k], square(at(theta, offset + k)));
    acc = acc === null ? term : add(acc, term);
  }
  return { value: acc, next: offset + consts.d };
}

/**
 * Build a kernel's matrix from packed hyperparameters, in the same order
 * `collectHypers` visits them.
 *
 * @param {Object} kernel - kernel instance (read for structure, not values)
 * @param {Object} consts - from {@link kernelConstants}
 * @param {Object} theta - grad Var holding the hyperparameters
 * @param {number} offset - index of this kernel's first hyperparameter
 * @returns {{K: Object, next: number}} matrix expression and the next offset
 */
export function kernelMatrixAD(kernel, consts, theta, offset) {
  const name = kernel.constructor.name;

  if (name === 'SumKernel') {
    let K = null;
    let cursor = offset;
    for (const child of kernel.kernels) {
      const part = kernelMatrixAD(child, consts, theta, cursor);
      K = K === null ? part.K : add(K, part.K);
      cursor = part.next;
    }
    return { K, next: cursor };
  }

  if (name === 'RBF') {
    const { value: sq, next } = scaledSquared(consts, theta, offset, kernel);
    const v = at(theta, next);
    return { K: mul(v, exp(mul(-0.5, sq))), next: next + 1 };
  }

  if (name === 'Matern') {
    const { value: sq, next } = scaledSquared(consts, theta, offset, kernel);
    const v = at(theta, next);
    const nu = kernel.nu;
    if (nu === Infinity) {
      return { K: mul(v, exp(mul(-0.5, sq))), next: next + 1 };
    }
    // sqrt(2*nu) * ||x - x'|| / l, with the length scales already folded in.
    //
    // The floor is load-bearing. `sq` is exactly 0 on the diagonal — a point's
    // distance to itself — and d(sqrt)/dx is infinite there, so the reverse
    // pass produced Infinity, met the diagonal's own zero coming the other way,
    // and returned NaN for every length-scale gradient while the VALUE stayed
    // exact. (The hand-derived path sidesteps this with an explicit
    // `if (scaledSq === 0) continue`.) A floor of 1e-30 makes the derivative
    // finite at 5e14 and shifts the kernel by ~1e-30 relative, far under
    // double precision.
    const sc = mul(Math.sqrt(2 * nu), sqrt(add(sq, MATERN_SQRT_FLOOR)));
    const decay = exp(mul(-1, sc));
    if (nu === 0.5) return { K: mul(v, decay), next: next + 1 };
    if (nu === 1.5) {
      return { K: mul(v, mul(add(1, sc), decay)), next: next + 1 };
    }
    if (nu === 2.5) {
      const poly = add(add(1, sc), mul(1 / 3, square(sc)));
      return { K: mul(v, mul(poly, decay)), next: next + 1 };
    }
    throw new Error(`autodiff kernel: Matern nu=${nu} is not supported (use 0.5, 1.5, 2.5 or Infinity)`);
  }

  if (name === 'RationalQuadratic') {
    const l = at(theta, offset);
    const v = at(theta, offset + 1);
    const a = at(theta, offset + 2);
    const term = add(1, div(consts.total, mul(2, mul(a, square(l)))));
    return { K: mul(v, powVar(term, a)), next: offset + 3 };
  }

  if (name === 'DotProduct') {
    const s0 = at(theta, offset);
    return { K: add(square(s0), consts.dot), next: offset + 1 };
  }

  if (name === 'ConstantKernel') {
    const c = at(theta, offset);
    const ones = Array.from({ length: consts.n }, () => new Array(consts.n).fill(1));
    return { K: mul(c, ones), next: offset + 1 };
  }

  if (name === 'WhiteKernel') {
    const level = at(theta, offset);
    const I = Array.from({ length: consts.n }, (_, i) =>
      Array.from({ length: consts.n }, (_, j) => (i === j ? 1 : 0)));
    return { K: mul(level, I), next: offset + 1 };
  }

  if (name === 'Periodic') {
    // The period is NOT a tunable hyperparameter here (collectHypers leaves
    // Periodic alone), so it enters as a constant.
    const l = at(theta, offset);
    const v = at(theta, offset + 1);
    const sinTerm = sinOfConst(consts.dist, kernel.period);
    return {
      K: mul(v, exp(div(mul(-2, square(sinTerm)), square(l)))),
      next: offset + 2,
    };
  }

  throw new Error(`autodiff kernel: ${name} has no differentiable form`);
}

/** sin(pi * dist / period) — constant, since the period is fixed. @private */
function sinOfConst(dist, period) {
  return dist.map((row) => row.map((r) => Math.sin((Math.PI * r) / period)));
}

/**
 * t^(-a) where the exponent is itself a hyperparameter. `pow` only takes a
 * constant power, so this goes through exp(-a·log t), which carries both
 * partials. t > 0 here by construction: it is 1 + a non-negative quantity.
 * @private
 */
function powVar(t, a) {
  return exp(mul(-1, mul(a, log(t))));
}

/**
 * Kernels this module can differentiate.
 *
 * RationalQuadratic needs box bounds to be safe here: its shape parameter is
 * degenerate upward — as alpha grows the kernel converges to an RBF — so the
 * likelihood has an unbounded ridge that an exact gradient will follow. Left
 * unbounded it ran alpha to 1e13 and took 24 s at n = 300. `collectHypers`
 * gives that parameter a `max`, and the optimizer runs in bounded space, which
 * is what makes it tractable. The derivative-free search never had the problem
 * only because it was too weak to find the ridge.
 * @private
 */
const SUPPORTED = new Set([
  'RBF', 'Matern', 'RationalQuadratic', 'DotProduct',
  'ConstantKernel', 'WhiteKernel', 'Periodic',
]);

/**
 * Can {@link kernelMatrixAD} build this kernel? Sums are supported when every
 * child is, and a Matérn only for the ν it has a closed form for.
 *
 * @param {Object} kernel
 * @returns {boolean}
 */
export function supportsAutodiff(kernel) {
  if (!kernel) return false;
  const name = kernel.constructor.name;
  if (name === 'SumKernel') return (kernel.kernels || []).every(supportsAutodiff);
  if (name === 'Matern') return [0.5, 1.5, 2.5, Infinity].includes(kernel.nu);
  return SUPPORTED.has(name);
}


/**
 * Per-dimension weights 1/l² for a stationary kernel, as a plain vector:
 * isotropic, ARD, or ARD by block. @private
 */
function inverseSquaredScales(kernel, d) {
  const l = kernel.lengthScale;
  const w = new Array(d);
  for (let k = 0; k < d; k++) {
    const lk = Array.isArray(l) ? l[kernel.blocks ? kernel.blocks[k] : k] : l;
    w[k] = 1 / (lk * lk);
  }
  return w;
}

/**
 * The cross-covariance vector k(x*, X) between ONE test point, held as a grad
 * `Var`, and the training inputs, held as constants. Differentiable in x*,
 * which is what a gradient-based search over the input needs: the predictive
 * mean is k(x*, X)·α and its derivative falls out of this.
 *
 * Hyperparameters are read from the kernel as numbers, since they are fixed
 * once the model is fitted. Supported: RBF, Matérn (any supported ν),
 * ConstantKernel, WhiteKernel (zero: a new point shares no noise with any
 * training point), and sums of those.
 *
 * @param {Object} kernel
 * @param {number[][]} XTrain - n × d
 * @param {Object} xVar - grad Var of shape [d]
 * @returns {Object} grad Var of shape [n]
 */
export function crossKernelAD(kernel, XTrain, xVar) {
  const name = kernel.constructor.name;
  const d = XTrain[0].length;

  if (name === 'SumKernel') {
    let acc = null;
    for (const child of kernel.kernels) {
      const part = crossKernelAD(child, XTrain, xVar);
      acc = acc === null ? part : add(acc, part);
    }
    return acc;
  }
  if (name === 'WhiteKernel') return mul(0, XTrain.map(() => 0));
  if (name === 'ConstantKernel') {
    const c = kernel.value !== undefined ? kernel.value : kernel.variance;
    return mul(c, XTrain.map(() => 1));
  }
  if (name !== 'RBF' && name !== 'Matern') {
    throw new Error(`crossKernelAD: ${name} is not supported (use RBF, Matern, WhiteKernel, ConstantKernel or a sum of them)`);
  }

  // x* broadcast to every training row, then the weighted squared distance
  // per row: (X - 1·x*ᵀ)² · w. grad broadcasts only a scalar, so the row
  // replication is an explicit rank-one product.
  const ones = XTrain.map(() => [1]);
  const Xs = matmul(ones, reshape(xVar, [1, d]));
  const sq = square(sub(XTrain, Xs));
  const scaledSq = matmul(sq, inverseSquaredScales(kernel, d));
  const v = kernel.variance;

  if (name === 'RBF' || kernel.nu === Infinity) return mul(v, exp(mul(-0.5, scaledSq)));
  const nu = kernel.nu;
  const sc = mul(Math.sqrt(2 * nu), sqrt(add(scaledSq, MATERN_SQRT_FLOOR)));
  const decay = exp(mul(-1, sc));
  if (nu === 0.5) return mul(v, decay);
  if (nu === 1.5) return mul(v, mul(add(1, sc), decay));
  if (nu === 2.5) return mul(v, mul(add(add(1, sc), mul(1 / 3, square(sc))), decay));
  throw new Error(`crossKernelAD: Matern nu=${nu} is not supported`);
}

/**
 * Is k(x, x) the same for every x? True for the stationary kernels above,
 * which is what lets the predictive variance's first term be a constant.
 * @param {Object} kernel
 * @returns {boolean}
 */
export function isStationary(kernel) {
  const name = kernel.constructor.name;
  if (name === 'SumKernel') return kernel.kernels.every(isStationary);
  return ['RBF', 'Matern', 'WhiteKernel', 'ConstantKernel'].includes(name);
}
