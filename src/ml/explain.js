/**
 * SHAP (SHapley Additive exPlanations) for @tangent.to/ds
 *
 * Model explanations based on Shapley values from cooperative game theory.
 * Every explainer is *additive*: for an instance x,
 *
 *   f(x) = baseValue + Σ_j φ_j(x)
 *
 * where `baseValue` is the model's expected output over a background/reference
 * distribution and `φ_j` is the contribution of feature j.
 *
 * Three explainers are provided:
 *
 *  - `KernelExplainer`  model-agnostic KernelSHAP (Lundberg & Lee, 2017).
 *                       Works with any model exposing a numeric `predict`,
 *                       e.g. a `GaussianProcessRegressor`, `MLPRegressor`, GLM…
 *  - `TreeExplainer`    exact, fast path-dependent TreeSHAP (Lundberg et al.,
 *                       2018) for `DecisionTreeRegressor` / `RandomForestRegressor`
 *                       (and bare `DecisionTreeBase`). Uses node coverage.
 *  - `PermutationExplainer`  model-agnostic Shapley estimation by sampling
 *                       feature permutations (Štrumbelj & Kononenko, 2014).
 *
 * Plus tidy-data helpers `summaryData` / `importanceData` that turn SHAP values
 * into row arrays ready for Observable Plot (beeswarm / bar).
 *
 * @module ml/explain
 */

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Coerce X into a 2-D array of numbers. A single row (1-D array) is wrapped. */
function _toRows(X) {
  if (!Array.isArray(X)) {
    throw new Error("explain: X must be an array (rows) or a single row array");
  }
  if (X.length === 0) return [];
  return Array.isArray(X[0]) ? X : [X];
}

/**
 * Resolve a numeric prediction function from a model or an explicit predictFn.
 * The returned function maps an array of rows to an array of numbers.
 *
 * @param {Object} opts - `{ model, predict }`
 * @returns {(rows:number[][]) => number[]}
 */
function _resolvePredict({ model, predict } = {}) {
  if (typeof predict === "function") {
    return (rows) => _asNumberArray(predict(rows));
  }
  if (model && typeof model.predict === "function") {
    return (rows) => _asNumberArray(model.predict(rows));
  }
  throw new Error(
    "explain: provide either a `predict` function or a `model` with a predict() method",
  );
}

/** Normalize a prediction result (array, {mean}, or tensor-like) to number[]. */
function _asNumberArray(out) {
  if (Array.isArray(out)) {
    // Could be array of numbers, or array of arrays (multi-output) -> take col 0
    if (out.length && Array.isArray(out[0])) return out.map((r) => r[0]);
    return out.map(Number);
  }
  if (out && Array.isArray(out.mean)) return out.mean.map(Number); // GP returnStd shape
  if (out && typeof out.arraySync === "function") {
    const a = out.arraySync();
    return Array.isArray(a[0]) ? a.map((r) => r[0]) : a.map(Number);
  }
  throw new Error("explain: prediction did not return a numeric array");
}

/** mean of a numeric array */
function _mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return a.length ? s / a.length : 0;
}

/** log of the binomial coefficient C(n, k), numerically stable */
function _logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return _logFact(n) - _logFact(k) - _logFact(n - k);
}
const _logFactCache = [0, 0];
function _logFact(n) {
  for (let i = _logFactCache.length; i <= n; i++) {
    _logFactCache[i] = _logFactCache[i - 1] + Math.log(i);
  }
  return _logFactCache[n];
}
function _choose(n, k) {
  return Math.round(Math.exp(_logChoose(n, k)));
}

/**
 * Mulberry32 — small deterministic PRNG so explanations are reproducible.
 * @param {number} seed
 */
function _rng(seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Solve the linear system A·x = b for a small dense symmetric system via
 * Gaussian elimination with partial pivoting. A is n×n, b is length n.
 * Returns x (length n). Falls back to a ridge if singular.
 */
function _solve(A, b) {
  const n = b.length;
  // Work on copies
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // pivot
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) {
      M[col][col] += 1e-8; // tiny ridge for numerical stability
      piv = col;
    }
    [M[col], M[piv]] = [M[piv], M[col]];
    const pivVal = M[col][col];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / pivVal;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  const x = new Array(n);
  for (let i = 0; i < n; i++) x[i] = M[i][n] / M[i][i];
  return x;
}

// ---------------------------------------------------------------------------
// KernelExplainer — model-agnostic KernelSHAP
// ---------------------------------------------------------------------------

/**
 * Model-agnostic SHAP via the KernelSHAP weighted-linear-regression estimator.
 *
 * The "absent" features in a coalition are marginalised by substituting values
 * from a background dataset and averaging the model output (the interventional
 * expectation). Keep the background small (a sample or summary, ~20–100 rows)
 * for performance: cost ≈ nCoalitions × nBackground model evaluations.
 *
 * @example
 * const ex = new KernelExplainer({ model: gp, background: Xref });
 * const { values, baseValue } = ex.shapValues(Xtest);
 * // values[i][j] is feature j's contribution for instance i
 */
export class KernelExplainer {
  /**
   * @param {Object} opts
   * @param {Object} [opts.model] - Fitted model with a numeric `predict(rows)`.
   * @param {Function} [opts.predict] - Alternatively, a predict function.
   * @param {number[][]} opts.background - Reference rows used to marginalise
   *   absent features. Its mean prediction is the explanation `baseValue`.
   * @param {string[]} [opts.featureNames] - Optional feature labels.
   */
  constructor({ model, predict, background, featureNames } = {}) {
    if (!background || !background.length) {
      throw new Error("KernelExplainer: a non-empty `background` dataset is required");
    }
    this._predict = _resolvePredict({ model, predict });
    this.background = _toRows(background);
    this.nFeatures = this.background[0].length;
    this.featureNames =
      featureNames || Array.from({ length: this.nFeatures }, (_, i) => `feature_${i}`);
    // Base value = expected model output over the background.
    this.expectedValue = _mean(this._predict(this.background));
  }

  /**
   * Compute SHAP values.
   *
   * @param {number[][]|number[]} X - Instance(s) to explain.
   * @param {Object} [opts]
   * @param {number|"auto"} [opts.nSamples="auto"] - Coalitions to evaluate when
   *   the feature count is large enough to require sampling. "auto" picks
   *   `2*M + 2048`. When `2^M` is below `maxExact`, all coalitions are used.
   * @param {number} [opts.maxExact=14] - Exhaustively enumerate coalitions when
   *   `nFeatures <= maxExact`.
   * @param {number} [opts.seed=0] - PRNG seed for the sampling path.
   * @returns {{ values:number[][], baseValue:number, expectedValue:number,
   *            featureNames:string[] }}
   */
  shapValues(X, opts = {}) {
    const rows = _toRows(X);
    const values = rows.map((x) => this._explainRow(x, opts));
    return {
      values,
      baseValue: this.expectedValue,
      expectedValue: this.expectedValue,
      featureNames: this.featureNames,
    };
  }

  /** Explain a single instance -> φ vector (length nFeatures). @private */
  _explainRow(x, { nSamples = "auto", maxExact = 14, seed = 0 } = {}) {
    const M = this.nFeatures;
    const fx = _mean(this._predict([x])); // single-row prediction
    const fnull = this.expectedValue;

    if (M === 1) return [fx - fnull];

    // Build the set of coalitions (subset masks) with kernel weights.
    const { masks, weights } = this._buildCoalitions(M, nSamples, maxExact, seed);

    // Predicted value for each coalition by substituting absent features from
    // the background and averaging.
    const yhat = masks.map((mask) => this._maskedPrediction(x, mask));

    return this._weightedShapSolve(masks, weights, yhat, fx, fnull, M);
  }

  /**
   * Average model output where present features (mask=1) take x's value and
   * absent features (mask=0) take each background row's value. @private
   */
  _maskedPrediction(x, mask) {
    const M = this.nFeatures;
    const synth = this.background.map((bg) => {
      const row = new Array(M);
      for (let j = 0; j < M; j++) row[j] = mask[j] ? x[j] : bg[j];
      return row;
    });
    return _mean(this._predict(synth));
  }

  /**
   * Enumerate or sample coalitions (excluding empty/full sets, which are pinned
   * by the efficiency constraint) with SHAP kernel weights. @private
   */
  _buildCoalitions(M, nSamples, maxExact, seed) {
    const masks = [];
    const weights = [];
    const kernelW = (s) =>
      (M - 1) / (_choose(M, s) * s * (M - s)); // SHAP kernel for |z|=s

    if (M <= maxExact) {
      // Exhaustive: every non-trivial subset.
      for (let bits = 1; bits < (1 << M) - 1; bits++) {
        const mask = new Array(M);
        let s = 0;
        for (let j = 0; j < M; j++) {
          mask[j] = (bits >> j) & 1;
          s += mask[j];
        }
        masks.push(mask);
        weights.push(kernelW(s));
      }
      return { masks, weights };
    }

    // Sampling path: enumerate whole subset-size layers from both ends inward
    // while budget allows, then random-sample the remaining layer.
    const budget = nSamples === "auto" ? 2 * M + 2048 : nSamples;
    const rand = _rng(seed);
    let used = 0;
    const seen = new Set();
    const addMask = (mask, w) => {
      const key = mask.join("");
      if (seen.has(key)) return false;
      seen.add(key);
      masks.push(mask);
      weights.push(w);
      return true;
    };
    const sizes = [];
    for (let s = 1; s <= Math.floor(M / 2); s++) {
      sizes.push(s);
      if (M - s !== s) sizes.push(M - s);
    }
    for (const s of sizes) {
      const layerCount = _choose(M, s);
      if (used + layerCount <= budget) {
        // enumerate the whole layer
        for (const idxs of _combinations(M, s)) {
          const mask = new Array(M).fill(0);
          idxs.forEach((j) => (mask[j] = 1));
          if (addMask(mask, kernelW(s))) used++;
        }
      } else {
        // random sample remaining budget from this layer, reweighting
        const take = Math.max(0, budget - used);
        const scaled = kernelW(s) * (layerCount / Math.max(1, take));
        let tries = 0;
        while (used < budget && tries < take * 20) {
          tries++;
          const mask = new Array(M).fill(0);
          let placed = 0;
          while (placed < s) {
            const j = Math.floor(rand() * M);
            if (!mask[j]) {
              mask[j] = 1;
              placed++;
            }
          }
          if (addMask(mask, scaled)) used++;
        }
        break;
      }
    }
    return { masks, weights };
  }

  /**
   * Weighted least squares for φ with the efficiency constraint
   * Σφ = fx − fnull, solved by eliminating the last feature (SHAP convention).
   * @private
   */
  _weightedShapSolve(masks, weights, yhat, fx, fnull, M) {
    const last = M - 1;
    const target = fx - fnull;
    // Design matrix etmp[s][j] = mask[j] - mask[last]  (j = 0..M-2)
    // Response  ey[s] = (yhat - fnull) - mask[last]*target
    const A = Array.from({ length: M - 1 }, () => new Array(M - 1).fill(0));
    const b = new Array(M - 1).fill(0);
    for (let s = 0; s < masks.length; s++) {
      const mask = masks[s];
      const w = weights[s];
      const ey = yhat[s] - fnull - mask[last] * target;
      const e = new Array(M - 1);
      for (let j = 0; j < M - 1; j++) e[j] = mask[j] - mask[last];
      for (let j = 0; j < M - 1; j++) {
        b[j] += w * e[j] * ey;
        for (let k = 0; k < M - 1; k++) A[j][k] += w * e[j] * e[k];
      }
    }
    const phiHead = _solve(A, b);
    const phi = new Array(M);
    let sumHead = 0;
    for (let j = 0; j < M - 1; j++) {
      phi[j] = phiHead[j];
      sumHead += phiHead[j];
    }
    phi[last] = target - sumHead; // efficiency constraint pins the last feature
    return phi;
  }
}

/** Generator of all index-combinations C(M, s). @private */
function* _combinations(M, s) {
  const idx = Array.from({ length: s }, (_, i) => i);
  if (s === 0) {
    yield [];
    return;
  }
  while (true) {
    yield idx.slice();
    let i = s - 1;
    while (i >= 0 && idx[i] === M - s + i) i--;
    if (i < 0) return;
    idx[i]++;
    for (let k = i + 1; k < s; k++) idx[k] = idx[k - 1] + 1;
  }
}

// ---------------------------------------------------------------------------
// TreeExplainer — exact path-dependent TreeSHAP
// ---------------------------------------------------------------------------

/**
 * Exact SHAP for tree models using the path-dependent algorithm of
 * Lundberg et al. (2018). Runs in O(T·L·D²) and uses each node's training
 * coverage (`nSamples`) as the conditional distribution.
 *
 * Supports regression trees: `DecisionTreeRegressor`, `RandomForestRegressor`,
 * a bare `DecisionTreeBase`, or any object exposing a compatible root node.
 *
 * @example
 * const ex = new TreeExplainer({ model: forest });
 * const { values, baseValue } = ex.shapValues(Xtest);
 */
export class TreeExplainer {
  /**
   * @param {Object} opts
   * @param {Object} opts.model - Fitted tree or forest regressor.
   * @param {string[]} [opts.featureNames]
   */
  constructor({ model, featureNames } = {}) {
    this.roots = _extractTreeRoots(model);
    if (!this.roots.length) {
      throw new Error(
        "TreeExplainer: could not find tree(s) on the model. Supported: " +
          "DecisionTreeRegressor, RandomForestRegressor, DecisionTreeBase.",
      );
    }
    this.nFeatures = _inferNFeatures(this.roots);
    this.featureNames =
      featureNames || Array.from({ length: this.nFeatures }, (_, i) => `feature_${i}`);
    // Per-tree expected value (coverage-weighted mean leaf value).
    this._treeBase = this.roots.map((r) => _treeExpectedValue(r));
    this.expectedValue = _mean(this._treeBase);
  }

  /**
   * @param {number[][]|number[]} X
   * @returns {{ values:number[][], baseValue:number, expectedValue:number,
   *            featureNames:string[] }}
   */
  shapValues(X) {
    const rows = _toRows(X);
    const M = this.nFeatures;
    const values = rows.map((x) => {
      const phi = new Array(M).fill(0);
      for (const root of this.roots) {
        const treePhi = _treeShapSingle(root, x, M);
        for (let j = 0; j < M; j++) phi[j] += treePhi[j];
      }
      for (let j = 0; j < M; j++) phi[j] /= this.roots.length; // forest = mean
      return phi;
    });
    return {
      values,
      baseValue: this.expectedValue,
      expectedValue: this.expectedValue,
      featureNames: this.featureNames,
    };
  }
}

/** Pull root node(s) out of supported tree/forest estimators. @private */
function _extractTreeRoots(model) {
  if (!model) return [];
  // Ensemble: array of tree wrappers, possibly behind a `.forest` wrapper.
  const treeList =
    (Array.isArray(model.trees) && model.trees) ||
    (model.forest && Array.isArray(model.forest.trees) && model.forest.trees) ||
    (Array.isArray(model.estimators_) && model.estimators_) ||
    null;
  if (treeList && treeList.length) {
    return treeList.map((t) => _rootOf(t)).filter(Boolean);
  }
  const root = _rootOf(model);
  return root ? [root] : [];
}

/** Get a root node from a tree estimator, wrapper, or raw node. @private */
function _rootOf(t) {
  if (!t) return null;
  if (t.type === "internal" || t.type === "leaf") return t; // already a node
  if (t.root) return t.root; // DecisionTreeBase
  if (t.tree && t.tree.root) return t.tree.root; // DecisionTreeRegressor wrapper
  return null;
}

/** Largest feature index used anywhere in the tree(s), +1. @private */
function _inferNFeatures(roots) {
  let max = -1;
  const visit = (n) => {
    if (!n || n.type === "leaf") return;
    if (n.feature > max) max = n.feature;
    visit(n.left);
    visit(n.right);
  };
  roots.forEach(visit);
  return max + 1;
}

/** Coverage-weighted mean leaf value = E[f] for one tree. @private */
function _treeExpectedValue(root) {
  let total = 0;
  const total0 = root.nSamples;
  const visit = (n) => {
    if (n.type === "leaf") {
      total += (n.nSamples / total0) * n.value;
      return;
    }
    visit(n.left);
    visit(n.right);
  };
  visit(root);
  return total;
}

/**
 * Path-dependent TreeSHAP for a single tree -> φ vector.
 * Implements Algorithm 2 of Lundberg et al. (2018). @private
 */
function _treeShapSingle(root, x, M) {
  const phi = new Array(M).fill(0);

  // A path element: feature index d, "zero" fraction z, "one" fraction o, weight w.
  function extend(path, pathLen, pz, po, pi) {
    const m = path.slice(0, pathLen).map((e) => ({ ...e }));
    m[pathLen] = { d: pi, z: pz, o: po, w: pathLen === 0 ? 1 : 0 };
    const l = pathLen;
    for (let i = l - 1; i >= 0; i--) {
      m[i + 1].w += (po * m[i].w * (i + 1)) / (l + 1);
      m[i].w = (pz * m[i].w * (l - i)) / (l + 1);
    }
    return m;
  }

  function unwind(path, i) {
    const l = path.length - 1;
    let n = path[l].w;
    const m = path.slice(0, l).map((e) => ({ ...e })); // length l
    const po = path[i].o;
    const pz = path[i].z;
    for (let j = l - 1; j >= 0; j--) {
      if (po !== 0) {
        const t = m[j].w;
        m[j].w = (n * (l + 1)) / ((j + 1) * po);
        n = t - (m[j].w * pz * (l - j)) / (l + 1);
      } else {
        m[j].w = (m[j].w * (l + 1)) / (pz * (l - j));
      }
    }
    // remove element i by shifting d/z/o down (weights already recomputed)
    for (let j = i; j < l; j++) {
      m[j].d = path[j + 1].d;
      m[j].z = path[j + 1].z;
      m[j].o = path[j + 1].o;
    }
    return m;
  }

  function unwoundSum(path, i) {
    const l = path.length - 1;
    const po = path[i].o;
    const pz = path[i].z;
    let total = 0;
    let n = path[l].w;
    if (po !== 0) {
      for (let j = l - 1; j >= 0; j--) {
        const tmp = (n * (l + 1)) / ((j + 1) * po);
        total += tmp;
        n = path[j].w - (tmp * pz * (l - j)) / (l + 1);
      }
    } else {
      for (let j = l - 1; j >= 0; j--) {
        total += (path[j].w / (pz * (l - j))) * (l + 1);
      }
    }
    return total;
  }

  function recurse(node, path, pathLen, pz, po, pi) {
    const m = extend(path, pathLen, pz, po, pi);
    const len = pathLen + 1;
    if (node.type === "leaf") {
      for (let i = 1; i < len; i++) {
        const w = unwoundSum(m.slice(0, len), i);
        phi[m[i].d] += w * (m[i].o - m[i].z) * node.value;
      }
      return;
    }
    const goLeft = x[node.feature] <= node.threshold;
    const hot = goLeft ? node.left : node.right;
    const cold = goLeft ? node.right : node.left;
    let iz = 1;
    let io = 1;
    // If this feature already appears on the path, unwind it first.
    let k = -1;
    for (let i = 1; i < len; i++) {
      if (m[i].d === node.feature) {
        k = i;
        break;
      }
    }
    let mUse = m.slice(0, len);
    let useLen = len;
    if (k !== -1) {
      iz = m[k].z;
      io = m[k].o;
      mUse = unwind(m.slice(0, len), k);
      useLen = len - 1;
    }
    const covNode = node.nSamples;
    recurse(
      hot,
      mUse,
      useLen,
      (iz * hot.nSamples) / covNode,
      io,
      node.feature,
    );
    recurse(
      cold,
      mUse,
      useLen,
      (iz * cold.nSamples) / covNode,
      0,
      node.feature,
    );
  }

  // initial path has capacity; start empty
  recurse(root, [], 0, 1, 1, -1);
  return phi;
}

// ---------------------------------------------------------------------------
// PermutationExplainer — Shapley estimation by permutation sampling
// ---------------------------------------------------------------------------

/**
 * Model-agnostic SHAP by sampling random feature orderings and accumulating
 * each feature's marginal contribution as it is "turned on" (its value swapped
 * from a background row to the instance's value). Uses antithetic pairs
 * (a permutation and its reverse) to reduce variance. Cheaper than KernelSHAP
 * for many features; converges to exact Shapley values as nPermutations grows.
 *
 * @example
 * const ex = new PermutationExplainer({ model: gp, background: Xref });
 * const { values, baseValue } = ex.shapValues(Xtest, { nPermutations: 64 });
 */
export class PermutationExplainer {
  /**
   * @param {Object} opts - `{ model | predict, background, featureNames }`
   */
  constructor({ model, predict, background, featureNames } = {}) {
    if (!background || !background.length) {
      throw new Error("PermutationExplainer: a non-empty `background` is required");
    }
    this._predict = _resolvePredict({ model, predict });
    this.background = _toRows(background);
    this.nFeatures = this.background[0].length;
    this.featureNames =
      featureNames || Array.from({ length: this.nFeatures }, (_, i) => `feature_${i}`);
    this.expectedValue = _mean(this._predict(this.background));
  }

  /**
   * @param {number[][]|number[]} X
   * @param {Object} [opts]
   * @param {number} [opts.nPermutations=64] - Antithetic permutation pairs.
   * @param {number} [opts.seed=0]
   * @returns {{ values:number[][], baseValue:number, expectedValue:number,
   *            featureNames:string[] }}
   */
  shapValues(X, { nPermutations = 64, seed = 0 } = {}) {
    const rows = _toRows(X);
    const rand = _rng(seed);
    const values = rows.map((x) => this._explainRow(x, nPermutations, rand));
    return {
      values,
      baseValue: this.expectedValue,
      expectedValue: this.expectedValue,
      featureNames: this.featureNames,
    };
  }

  /** @private */
  _explainRow(x, nPermutations, rand) {
    const M = this.nFeatures;
    const phi = new Array(M).fill(0);
    let count = 0;
    for (let p = 0; p < nPermutations; p++) {
      const perm = _shuffle(
        Array.from({ length: M }, (_, i) => i),
        rand,
      );
      // antithetic pair: forward and reversed ordering
      this._accumulate(x, perm, phi);
      this._accumulate(x, perm.slice().reverse(), phi);
      count += 2;
    }
    for (let j = 0; j < M; j++) phi[j] /= count;
    return phi;
  }

  /**
   * One ordering, averaged over background rows: start from a background row and
   * swap in x's features in `perm` order, crediting each step's delta. @private
   */
  _accumulate(x, perm, phi) {
    const M = this.nFeatures;
    const bg = this.background;
    const nb = bg.length;
    // Build the sequence of M+1 synthetic rows per background sample at once
    // by evaluating predictions in batches for efficiency.
    const rows = [];
    for (let b = 0; b < nb; b++) {
      const cur = bg[b].slice();
      rows.push(cur.slice()); // step 0: full background
      for (let s = 0; s < M; s++) {
        cur[perm[s]] = x[perm[s]];
        rows.push(cur.slice());
      }
    }
    const preds = this._predict(rows);
    // preds layout: per background sample, (M+1) consecutive predictions
    for (let b = 0; b < nb; b++) {
      const off = b * (M + 1);
      for (let s = 0; s < M; s++) {
        phi[perm[s]] += (preds[off + s + 1] - preds[off + s]) / nb;
      }
    }
  }
}

/** Fisher–Yates shuffle with an injected PRNG. @private */
function _shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------------------------------------------------------------------------
// Functional shortcuts
// ---------------------------------------------------------------------------

/**
 * Convenience: KernelSHAP in one call. See {@link KernelExplainer}.
 * @param {Object} spec - Explainer specification
 * @param {Object} spec.model - Model object to explain (used when no predict function is supplied)
 * @param {Function} spec.predict - Prediction function mapping instances to outputs
 * @param {Array<Array<number>>} spec.background - Background/reference dataset (n × p)
 * @param {Array<string>} spec.featureNames - Feature names for each column
 * @param {Array<Array<number>>} X - Instances to explain (n × p)
 * @param {Object} opts - Options forwarded to KernelExplainer.shapValues
 * @returns {{values: Array<Array<number>>, featureNames?: Array<string>}} SHAP values per instance and feature
 */
export function kernelShap({ model, predict, background, featureNames }, X, opts) {
  return new KernelExplainer({ model, predict, background, featureNames }).shapValues(
    X,
    opts,
  );
}

/**
 * Convenience: TreeSHAP in one call. See {@link TreeExplainer}.
 * @param {Object} spec - Explainer specification
 * @param {Object} spec.model - Tree model to explain
 * @param {Array<string>} spec.featureNames - Feature names for each column
 * @param {Array<Array<number>>} X - Instances to explain (n × p)
 * @returns {{values: Array<Array<number>>, featureNames?: Array<string>}} SHAP values per instance and feature
 */
export function treeShap({ model, featureNames }, X) {
  return new TreeExplainer({ model, featureNames }).shapValues(X);
}

// ---------------------------------------------------------------------------
// Plotting / tidy-data helpers (ready for Observable Plot)
// ---------------------------------------------------------------------------

/**
 * Global feature importance = mean(|SHAP|) per feature, sorted descending.
 * Plug into `Plot.barX(importanceData(res), { x: "importance", y: "feature" })`.
 *
 * @param {{values:number[][], featureNames?:string[]}} res - shapValues() output.
 * @returns {Array<{feature:string, importance:number}>}
 */
export function importanceData(res) {
  const { values, featureNames } = res;
  const M = values[0]?.length || 0;
  const names = featureNames || Array.from({ length: M }, (_, i) => `feature_${i}`);
  const imp = new Array(M).fill(0);
  for (const row of values) {
    for (let j = 0; j < M; j++) imp[j] += Math.abs(row[j]);
  }
  return imp
    .map((v, j) => ({ feature: names[j], importance: v / Math.max(1, values.length) }))
    .sort((a, b) => b.importance - a.importance);
}

/**
 * Tidy long-form rows for a beeswarm / summary plot: one row per
 * (instance, feature) carrying the SHAP value and the original feature value.
 * Plug into `Plot.dot(summaryData(res, X), { x: "shap", y: "feature", fill: "featureValue" })`.
 *
 * @param {{values:number[][], featureNames?:string[]}} res - shapValues() output.
 * @param {number[][]} X - The explained instances (same order as res.values).
 * @returns {Array<{instance:number, feature:string, featureIndex:number,
 *                  shap:number, featureValue:number}>}
 */
export function summaryData(res, X) {
  const { values, featureNames } = res;
  const rows = _toRows(X);
  const M = values[0]?.length || 0;
  const names = featureNames || Array.from({ length: M }, (_, i) => `feature_${i}`);
  const out = [];
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < M; j++) {
      out.push({
        instance: i,
        feature: names[j],
        featureIndex: j,
        shap: values[i][j],
        featureValue: rows[i] ? rows[i][j] : null,
      });
    }
  }
  return out;
}

export default {
  KernelExplainer,
  TreeExplainer,
  PermutationExplainer,
  kernelShap,
  treeShap,
  importanceData,
  summaryData,
};
