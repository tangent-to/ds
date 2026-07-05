/**
 * Gaussian Process Regressor
 * 
 * A scikit-learn style Gaussian Process implementation.
 * Supports:
 * - Fitting to training data
 * - Prediction with uncertainty quantification
 * - Sampling from posterior distribution
 * 
 * @example
 * const gp = new GaussianProcessRegressor({ kernel: 'rbf', lengthScale: 1.0 });
 * gp.fit(X_train, y_train);
 * const { mean, std } = gp.predict(X_test, { returnStd: true });
 * const samples = gp.sample(X_test, 5);
 */

import { Regressor } from '../../core/estimators/estimator.js';
import { toMatrix } from '../../core/linalg.js';
import { prepareXY } from '../../core/table.js';
import {
  Kernel,
  RBF,
  Periodic,
  RationalQuadratic,
  ConstantKernel,
  Matern,
} from '../kernels/index.js';

/**
 * Seeded random number generator (Mulberry32)
 * @param {number} seed - Seed value
 * @returns {Function} Random number generator function
 */
function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/**
 * Cholesky decomposition (lower triangular)
 * @param {Matrix} A - Symmetric positive definite matrix
 * @returns {Matrix} Lower triangular matrix L such that A = L * L^T
 */
function choleskyDecomposition(A) {
  const n = A.rows;
  const L = new Array(n);
  
  for (let i = 0; i < n; i++) {
    L[i] = new Array(n).fill(0);
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      
      if (j === i) {
        for (let k = 0; k < j; k++) {
          sum += L[j][k] * L[j][k];
        }
        const diag = A.get(j, j) - sum;
        if (diag <= 0) {
          throw new Error('Matrix is not positive definite');
        }
        L[j][j] = Math.sqrt(diag);
      } else {
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (A.get(i, j) - sum) / L[j][j];
      }
    }
  }

  return toMatrix(L);
}

/**
 * Sample from multivariate normal distribution
 * @param {Array<number>} mean - Mean vector
 * @param {Matrix} cov - Covariance matrix
 * @param {Function} rng - Random number generator
 * @returns {Array<number>} Sample
 */
function sampleMultivariateNormal(mean, cov, rng = Math.random) {
  const n = mean.length;
  let L;
  
  try {
    L = choleskyDecomposition(cov);
  } catch {
    // Add jitter if not positive definite
    const jitter = 1e-6;
    for (let i = 0; i < n; i++) {
      cov.set(i, i, cov.get(i, i) + jitter);
    }
    L = choleskyDecomposition(cov);
  }

  // Generate standard normal samples using Box-Muller
  const z = new Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = rng();
    const u2 = rng();
    z[i] = Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  // Transform: sample = mean + L * z
  const sample = new Array(n);
  for (let i = 0; i < n; i++) {
    sample[i] = mean[i];
    for (let j = 0; j <= i; j++) {
      sample[i] += L.get(i, j) * z[j];
    }
  }

  return sample;
}

/**
 * Collect the tunable positive hyperparameters of a kernel as get/set/min
 * handles, descending into SumKernel children. Length-scale arrays (ARD)
 * contribute one entry per dimension. @private
 */
function collectHypers(kernel) {
  const entries = [];
  const visit = (k) => {
    if (!k) return;
    const name = k.constructor.name;
    if (name === "SumKernel") {
      (k.kernels || []).forEach(visit);
      return;
    }
    if (name === "Matern" || name === "RBF") {
      if (Array.isArray(k.lengthScale)) {
        k.lengthScale.forEach((_, i) =>
          entries.push({ get: () => k.lengthScale[i], set: (v) => { k.lengthScale[i] = v; }, min: 1e-5 }),
        );
      } else {
        entries.push({ get: () => k.lengthScale, set: (v) => { k.lengthScale = v; }, min: 1e-5 });
      }
      entries.push({ get: () => k.variance, set: (v) => { k.variance = v; }, min: 1e-8 });
    } else if (name === "RationalQuadratic") {
      entries.push({ get: () => k.lengthScale, set: (v) => { k.lengthScale = v; }, min: 1e-5 });
      entries.push({ get: () => k.variance, set: (v) => { k.variance = v; }, min: 1e-8 });
      if (k.alpha !== undefined) {
        entries.push({ get: () => k.alpha, set: (v) => { k.alpha = v; }, min: 1e-5 });
      }
    } else if (name === "DotProduct") {
      entries.push({ get: () => k.sigma0, set: (v) => { k.sigma0 = v; }, min: 1e-8 });
    } else if (name === "ConstantKernel") {
      const key = k.value !== undefined ? "value" : "variance";
      entries.push({ get: () => k[key], set: (v) => { k[key] = v; }, min: 1e-8 });
    }
    // Other kernels (e.g. Periodic) are left fixed.
  };
  visit(kernel);
  return entries;
}

/**
 * Derivative-free coordinate pattern search in log-space. Minimizes `evalNeg`
 * by stepping each hyperparameter up/down by a shrinking factor. Robust for the
 * smooth, low-dimensional marginal-likelihood surface. @private
 *
 * @returns {{ vals:number[], f:number }} best raw values and objective.
 */
function _patternSearch(hypers, evalNeg) {
  const apply = (logVals) =>
    hypers.forEach((h, i) => h.set(Math.max(h.min, Math.exp(logVals[i]))));
  let x = hypers.map((h) => Math.log(h.get()));
  apply(x);
  let f = evalNeg();
  let step = 1.0; // a factor of e per step
  const minStep = 0.05;
  const maxIter = 300;
  let it = 0;
  while (step > minStep && it < maxIter) {
    let improved = false;
    for (let d = 0; d < x.length; d++) {
      for (const s of [step, -step]) {
        const xt = x.slice();
        xt[d] += s;
        apply(xt);
        const ft = evalNeg();
        if (ft < f - 1e-9) {
          f = ft;
          x = xt;
          improved = true;
          break;
        }
      }
    }
    if (!improved) step *= 0.5;
    it++;
  }
  apply(x);
  return { vals: hypers.map((h) => h.get()), f };
}

export class GaussianProcessRegressor extends Regressor {
  /**
   * @param {Object} opts - Options
   * @param {Kernel|string} opts.kernel - Kernel instance or type ('rbf', 'periodic', 'rational_quadratic')
   * @param {number} opts.lengthScale - Length scale for kernel (default: 1.0)
   * @param {number} opts.variance - Signal variance (default: 1.0)
   * @param {number} opts.alpha - Noise level / regularization (default: 1e-10)
   * @param {number} opts.noiseLevel - Alias for alpha
   * @param {number} opts.period - Period for periodic kernel
   * @param {boolean} opts.normalizeY - Standardize the target (center + scale to
   *   unit variance) before fitting; predictions, std, covariance and posterior
   *   samples are back-transformed. Alias: `normalize_y` (default: false)
   */
  constructor(opts = {}) {
    super(opts);
    
    // Create kernel
    if (opts.kernel instanceof Kernel) {
      this.kernel = opts.kernel;
    } else {
      const kernelType = (opts.kernel || 'rbf').toLowerCase();
      const lengthScale = opts.lengthScale ?? 1.0;
      const variance = opts.variance ?? opts.amplitude ?? 1.0;
      
      switch (kernelType) {
        case 'rbf':
          this.kernel = new RBF(lengthScale, variance);
          break;
        case 'periodic':
          this.kernel = new Periodic(lengthScale, opts.period || 1.0, variance);
          break;
        case 'rational_quadratic':
        case 'rationalquadratic':
          this.kernel = new RationalQuadratic(lengthScale, opts.alpha || 1.0, variance);
          break;
        case 'matern':
          this.kernel = new Matern({ lengthScale, nu: opts.nu ?? 1.5, amplitude: variance });
          break;
        case 'constant':
        case 'constantkernel':
          this.kernel = new ConstantKernel({ value: variance });
          break;
        default:
          throw new Error(`Unknown kernel type: ${kernelType}`);
      }
    }
    
    // Support both alpha and noiseLevel
    this.alpha = opts.alpha ?? opts.noiseLevel ?? 1e-10;

    // Optionally standardize the target before fitting (sklearn `normalize_y`).
    // A GP has a zero-mean prior, so on a target with a large mean the posterior
    // reverts toward 0 away from the training data and the kernel amplitude must
    // absorb the offset. Centring (and scaling) y fixes both; predictions,
    // std, covariance and posterior samples are back-transformed automatically.
    this.normalizeY = opts.normalizeY ?? opts.normalize_y ?? false;
    this._yMean = 0;
    this._yStd = 1;

    // Hyperparameter optimization (maximize the log marginal likelihood).
    // Off by default for backward compatibility; opt in with `optimize: true`
    // (or sklearn-style `optimizer`/`nRestartsOptimizer`). When enabled, fit()
    // tunes the kernel length scale(s) + variance and the noise `alpha`.
    this.optimize =
      opts.optimize === true ||
      (opts.optimizer !== undefined && opts.optimizer !== false && opts.optimizer !== null);
    this.nRestarts = opts.nRestarts ?? opts.nRestartsOptimizer ?? 0;
    this._seed = opts.randomState ?? opts.seed ?? 42;

    // Internal state
    this._XTrain = null;
    this._yTrain = null;
    this._L = null;
    this._alphaVector = null;
    this.logMarginalLikelihood_ = null;
  }

  /**
   * Fit the GP to training data
   * @param {Array<Array<number>>|Object} X - Training inputs (n samples × d
   *   features), or a declarative spec `{ X, columns, y, data, omit_missing }`
   * @param {Array<number>} [y] - Training targets (n)
   * @returns {this} The fitted estimator (for chaining)
   */
  fit(X, y = null) {
    // Handle declarative input
    let dataX, dataY;
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.X)) {
      const prepared = prepareXY({
        X: X.X || X.columns,
        y: X.y,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      dataX = prepared.X;
      dataY = prepared.y;
    } else {
      dataX = X;
      dataY = y;
    }

    // Convert to matrix
    this._XTrain = toMatrix(dataX);
    const yRaw = Array.isArray(dataY) ? [...dataY] : Array.from(dataY);

    // Standardize the target if requested; factorization below works on the
    // transformed y, and predict()/sample() invert the transform.
    if (this.normalizeY) {
      const m = yRaw.reduce((a, b) => a + b, 0) / yRaw.length;
      const v = yRaw.reduce((a, b) => a + (b - m) ** 2, 0) / yRaw.length;
      this._yMean = m;
      this._yStd = v > 0 ? Math.sqrt(v) : 1;
      this._yTrain = yRaw.map((val) => (val - this._yMean) / this._yStd);
    } else {
      this._yMean = 0;
      this._yStd = 1;
      this._yTrain = yRaw;
    }

    // Optionally tune hyperparameters by maximizing the log marginal likelihood.
    if (this.optimize) {
      this._optimizeHypers();
    }

    // Factorize with the final hyperparameters.
    this._refit();
    this.logMarginalLikelihood_ = -this._negLogML();

    this.fitted = true;
    return this;
  }

  /**
   * Compute K + αI, its Cholesky factor, and the solve vector α = K⁻¹y.
   * Uses the current kernel hyperparameters and noise. @private
   */
  _refit() {
    const K = this.kernel.call(this._XTrain);
    for (let i = 0; i < K.rows; i++) {
      K.set(i, i, K.get(i, i) + this.alpha);
    }
    try {
      this._L = choleskyDecomposition(K);
    } catch (error) {
      throw new Error(`Failed to fit GP: ${error.message}. Try increasing alpha.`);
    }
    this._alphaVector = this._solveCholesky(this._L, this._yTrain);
    return this;
  }

  /**
   * Log marginal likelihood of the training data under the current
   * hyperparameters: log p(y|X) = -½ yᵀK⁻¹y - ½ log|K| - n/2 log(2π).
   * Requires the model to have seen training data (via fit).
   * @returns {number}
   */
  logMarginalLikelihood() {
    if (!this._XTrain) {
      throw new Error("logMarginalLikelihood() requires training data; call fit() first.");
    }
    return -this._negLogML();
  }

  /**
   * Negative log marginal likelihood for the current kernel + alpha.
   * Returns a large finite penalty if K is not positive definite. @private
   */
  _negLogML() {
    const K = this.kernel.call(this._XTrain);
    const n = K.rows;
    for (let i = 0; i < n; i++) {
      K.set(i, i, K.get(i, i) + this.alpha);
    }
    let L;
    try {
      L = choleskyDecomposition(K);
    } catch {
      return 1e12; // not PD under these hypers -> heavy penalty
    }
    const alphaVec = this._solveCholesky(L, this._yTrain);
    let yAlpha = 0;
    for (let i = 0; i < n; i++) yAlpha += this._yTrain[i] * alphaVec[i];
    let logDet = 0; // ½ log|K| = Σ log L_ii
    for (let i = 0; i < n; i++) logDet += Math.log(L.get(i, i));
    const logML = -0.5 * yAlpha - logDet - 0.5 * n * Math.log(2 * Math.PI);
    return -logML;
  }

  /**
   * Maximize the log marginal likelihood over kernel length scale(s),
   * variance(s) and the noise `alpha`, by a derivative-free log-space pattern
   * search with optional random restarts. Mutates the kernel and `this.alpha`.
   * @private
   */
  _optimizeHypers() {
    const hypers = collectHypers(this.kernel);
    // Treat the observation noise as a (WhiteKernel-like) hyperparameter too.
    hypers.push({ get: () => this.alpha, set: (v) => { this.alpha = v; }, min: 1e-10 });
    if (hypers.length === 0) return;

    const initial = hypers.map((h) => h.get());
    const evalNeg = () => this._negLogML();
    const rng = mulberry32(this._seed);

    let bestVals = initial.slice();
    let bestF = evalNeg();

    for (let r = 0; r <= this.nRestarts; r++) {
      if (r === 0) {
        hypers.forEach((h, i) => h.set(initial[i]));
      } else {
        // Random restart: perturb each hyperparameter in log-space.
        hypers.forEach((h, i) => {
          const factor = Math.exp((rng() * 2 - 1) * 2.0); // ×[e⁻², e²]
          h.set(Math.max(h.min, initial[i] * factor));
        });
      }
      const { vals, f } = _patternSearch(hypers, evalNeg);
      if (f < bestF) {
        bestF = f;
        bestVals = vals;
      }
    }
    hypers.forEach((h, i) => h.set(bestVals[i]));
  }

  /**
   * Predict at test points
   * @param {Array<Array<number>>} X - Test inputs (m samples × d features)
   * @param {Object} [opts] - Options
   * @param {boolean} [opts.returnStd] - Return per-point standard deviations
   * @param {boolean} [opts.returnCov] - Return the full posterior covariance
   * @returns {Array<number>|{mean: Array<number>, std?: Array<number>, covariance?: Array<Array<number>>}}
   *   Predicted means, or an object with mean and std/covariance when requested
   */
  predict(X, opts = {}) {
    this._ensureFitted('predict');
    const { returnStd = false, returnCov = false } = opts;

    const XTest = toMatrix(X);
    const KStar = this.kernel.call(this._XTrain, XTest);

    // Compute mean: K* @ alpha (in standardized-y space), then back-transform.
    const mean = new Array(XTest.rows);
    for (let i = 0; i < XTest.rows; i++) {
      mean[i] = 0;
      for (let j = 0; j < this._XTrain.rows; j++) {
        mean[i] += KStar.get(j, i) * this._alphaVector[j];
      }
      mean[i] = mean[i] * this._yStd + this._yMean;
    }

    if (!returnStd && !returnCov) {
      return mean;
    }

    const { covarianceMatrix, diag } = this._computePosteriorCovariance(XTest, KStar);
    const result = { mean };

    if (returnStd) {
      // std scales with y; variance (diag) scales with y².
      result.std = diag.map(v => Math.sqrt(Math.max(0, v)) * this._yStd);
    }

    if (returnCov) {
      const s2 = this._yStd * this._yStd;
      result.covariance = covarianceMatrix.to2DArray().map(row => row.map(x => x * s2));
    }

    return result;
  }

  /**
   * Sample from the posterior distribution
   * @param {Array} X - Test inputs
   * @param {number} nSamples - Number of samples
   * @param {number} [seed] - Random seed for reproducibility
   * @returns {Array<Array>} Array of samples
   */
  sample(X, nSamples = 1, seed = null) {
    this._ensureFitted('sample');

    const rng = seed !== null ? mulberry32(seed) : Math.random;
    const XTest = toMatrix(X);
    const KStar = this.kernel.call(this._XTrain, XTest);

    // Mean of posterior
    const mean = new Array(XTest.rows);
    for (let i = 0; i < XTest.rows; i++) {
      mean[i] = 0;
      for (let j = 0; j < this._XTrain.rows; j++) {
        mean[i] += KStar.get(j, i) * this._alphaVector[j];
      }
    }

    const { covarianceMatrix } = this._computePosteriorCovariance(XTest, KStar);

    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      samples.push(sampleMultivariateNormal(mean, covarianceMatrix, rng));
    }

    // Draws are in standardized-y space; back-transform to the original scale.
    if (this.normalizeY) {
      return samples.map((row) => row.map((v) => v * this._yStd + this._yMean));
    }
    return samples;
  }

  /**
   * Sample from the prior (unfitted GP)
   * @param {Array} X - Input points
   * @param {number} nSamples - Number of samples
   * @param {number} [seed] - Random seed for reproducibility
   * @returns {Array<Array>} Array of samples
   */
  samplePrior(X, nSamples = 1, seed = null) {
    const rng = seed !== null ? mulberry32(seed) : Math.random;
    const XMatrix = toMatrix(X);
    const n = XMatrix.rows;
    
    // Compute kernel matrix
    const K = this.kernel.call(XMatrix);
    
    // Add small noise for numerical stability
    for (let i = 0; i < n; i++) {
      K.set(i, i, K.get(i, i) + 1e-10);
    }

    const mean = new Array(n).fill(0);
    
    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      samples.push(sampleMultivariateNormal(mean, K, rng));
    }

    return samples;
  }

  _computePosteriorCovariance(XTest, KStar) {
    const nTrain = this._XTrain.rows;
    const nTest = XTest.rows;

    // Solve L @ V = K*
    const V = new Array(nTrain);
    for (let i = 0; i < nTrain; i++) {
      V[i] = new Array(nTest).fill(0);
    }

    for (let col = 0; col < nTest; col++) {
      const kStarColumn = new Array(nTrain);
      for (let row = 0; row < nTrain; row++) {
        kStarColumn[row] = KStar.get(row, col);
      }
      const solved = this._forwardSubstitution(this._L, kStarColumn);
      for (let row = 0; row < nTrain; row++) {
        V[row][col] = solved[row];
      }
    }

    const covarianceMatrix = this.kernel.call(XTest);
    const diag = new Array(nTest);

    for (let i = 0; i < nTest; i++) {
      for (let j = 0; j <= i; j++) {
        let cov = covarianceMatrix.get(i, j);
        for (let k = 0; k < nTrain; k++) {
          cov -= V[k][i] * V[k][j];
        }
        covarianceMatrix.set(i, j, cov);
        if (i !== j) {
          covarianceMatrix.set(j, i, cov);
        }
        if (i === j) {
          diag[i] = cov;
        }
      }
    }

    return { covarianceMatrix, diag };
  }

  _solveCholesky(L, y) {
    const z = this._forwardSubstitution(L, y);
    return this._backSubstitution(L, z);
  }

  _forwardSubstitution(L, b) {
    const n = L.rows;
    const x = new Array(n);

    for (let i = 0; i < n; i++) {
      x[i] = b[i];
      for (let j = 0; j < i; j++) {
        x[i] -= L.get(i, j) * x[j];
      }
      x[i] /= L.get(i, i);
    }

    return x;
  }

  _backSubstitution(L, b) {
    const n = L.rows;
    const x = new Array(n);

    for (let i = n - 1; i >= 0; i--) {
      x[i] = b[i];
      for (let j = i + 1; j < n; j++) {
        x[i] -= L.get(j, i) * x[j];
      }
      x[i] /= L.get(i, i);
    }

    return x;
  }

  toJSON() {
    return {
      type: 'GaussianProcessRegressor',
      kernel: {
        type: this.kernel.constructor.name,
        params: this.kernel.getParams()
      },
      alpha: this.alpha,
      normalizeY: this.normalizeY,
      yMean: this._yMean,
      yStd: this._yStd,
      fitted: this.fitted,
      XTrain: this._XTrain ? this._XTrain.to2DArray() : null,
      yTrain: this._yTrain,
      L: this._L ? this._L.to2DArray() : null,
      alphaVector: this._alphaVector
    };
  }

  static fromJSON(json) {
    let kernel;
    switch (json.kernel.type) {
      case 'RBF':
        kernel = new RBF(json.kernel.params.lengthScale, json.kernel.params.variance);
        break;
      case 'Periodic':
        kernel = new Periodic(json.kernel.params.lengthScale, json.kernel.params.period, json.kernel.params.variance);
        break;
      case 'RationalQuadratic':
        kernel = new RationalQuadratic(json.kernel.params.lengthScale, json.kernel.params.alpha, json.kernel.params.variance);
        break;
      default:
        throw new Error(`Unknown kernel type: ${json.kernel.type}`);
    }

    const gp = new GaussianProcessRegressor({ kernel, alpha: json.alpha, normalizeY: json.normalizeY });
    gp._yMean = json.yMean ?? 0;
    gp._yStd = json.yStd ?? 1;

    if (json.fitted) {
      gp._XTrain = toMatrix(json.XTrain);
      gp._yTrain = json.yTrain;
      gp._L = toMatrix(json.L);
      gp._alphaVector = json.alphaVector;
      gp.fitted = true;
    }

    return gp;
  }
}
