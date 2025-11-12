/**
 * Multinomial Logistic Regression
 *
 * Fits K-1 coefficient vectors simultaneously for K classes
 * Uses iteratively reweighted least squares (IRLS) with Newton-Raphson
 * Matches R's nnet::multinom() behavior
 */

import { Matrix } from 'ml-matrix';
import { sum } from '../core/math.js';

/**
 * Softmax function with numerical stability
 * @param {Array<number>} logits - Linear predictors for each class
 * @returns {Array<number>} Probabilities that sum to 1
 */
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exp_logits = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = sum(exp_logits);
  return exp_logits.map((e) => e / sumExp);
}

/**
 * Compute multinomial log-likelihood
 * @param {Array<Array<number>>} X - Design matrix (n × p)
 * @param {Array<number>} y - Response vector (class indices 0, 1, ..., K-1)
 * @param {Array<Array<number>>} coefficients - K-1 × p matrix of coefficients
 * @param {number} K - Number of classes
 * @returns {number} Log-likelihood
 */
function computeLogLikelihood(X, y, coefficients, K) {
  const n = X.length;
  const p = X[0].length;
  let logLik = 0;

  for (let i = 0; i < n; i++) {
    // Compute logits for all K classes
    const logits = new Array(K);

    // Reference class has logit = 0
    logits[0] = 0;

    // Other K-1 classes
    for (let k = 1; k < K; k++) {
      let logit = 0;
      for (let j = 0; j < p; j++) {
        logit += coefficients[k - 1][j] * X[i][j];
      }
      logits[k] = logit;
    }

    // Compute probabilities via softmax
    const probs = softmax(logits);

    // Add log-likelihood for this observation
    const yi = y[i];
    logLik += Math.log(Math.max(probs[yi], 1e-15));
  }

  return logLik;
}

/**
 * Compute gradient and Hessian for multinomial logistic regression
 * @param {Array<Array<number>>} X - Design matrix (n × p)
 * @param {Array<number>} y - Response vector (class indices)
 * @param {Array<Array<number>>} coefficients - K-1 × p matrix
 * @param {number} K - Number of classes
 * @returns {Object} {gradient, hessian}
 */
function computeGradientHessian(X, y, coefficients, K) {
  const n = X.length;
  const p = X[0].length;

  // Flatten coefficient matrix into vector: [(K-1) × p] long
  const nParams = (K - 1) * p;
  const gradient = new Array(nParams).fill(0);

  // Hessian will be nParams × nParams
  const hessian = Array(nParams).fill(null).map(() => Array(nParams).fill(0));

  // For each observation
  for (let i = 0; i < n; i++) {
    // Compute logits and probabilities
    const logits = new Array(K);
    logits[0] = 0; // reference class

    for (let k = 1; k < K; k++) {
      let logit = 0;
      for (let j = 0; j < p; j++) {
        logit += coefficients[k - 1][j] * X[i][j];
      }
      logits[k] = logit;
    }

    const probs = softmax(logits);

    // Update gradient
    for (let k = 1; k < K; k++) {
      const yik = (y[i] === k) ? 1 : 0;
      const residual = yik - probs[k];

      for (let j = 0; j < p; j++) {
        const idx = (k - 1) * p + j;
        gradient[idx] += residual * X[i][j];
      }
    }

    // Update Hessian (observed information matrix)
    // H[idx1, idx2] = sum_i X[i,j1] * X[i,j2] * (prob[k1] * (I(k1==k2) - prob[k2]))
    for (let k1 = 1; k1 < K; k1++) {
      for (let k2 = 1; k2 < K; k2++) {
        const factor = probs[k1] * ((k1 === k2 ? 1 : 0) - probs[k2]);

        for (let j1 = 0; j1 < p; j1++) {
          for (let j2 = 0; j2 < p; j2++) {
            const idx1 = (k1 - 1) * p + j1;
            const idx2 = (k2 - 1) * p + j2;
            hessian[idx1][idx2] -= X[i][j1] * X[i][j2] * factor;
          }
        }
      }
    }
  }

  return { gradient, hessian };
}

/**
 * Fit multinomial logistic regression via Newton-Raphson
 * @param {Array<Array<number>>} X - Design matrix (n × p)
 * @param {Array<number>} y - Response vector (class indices 0, 1, ..., K-1)
 * @param {Object} options - Fitting options
 * @returns {Object} Fitted model
 */
export function fitMultinomial(X, y, options = {}) {
  const {
    intercept = true,
    maxIter = 100,
    tol = 1e-6,
    weights = null,
  } = options;

  const n = X.length;
  let p = X[0].length;

  // Add intercept if requested
  let Xmat = intercept ? X.map((row) => [1, ...row]) : X.map((row) => [...row]);

  if (intercept) p++;

  // Determine number of classes
  const K = Math.max(...y) + 1;

  if (K < 3) {
    throw new Error(
      'Multinomial requires at least 3 classes. Use binomial family for binary classification.',
    );
  }

  // Initialize coefficients: (K-1) × p matrix
  // Start with zeros
  let coefficients = Array(K - 1).fill(null).map(() => Array(p).fill(0));

  let converged = false;
  let iteration = 0;
  let logLik = -Infinity;

  for (iteration = 0; iteration < maxIter; iteration++) {
    const logLikPrev = logLik;

    // Compute gradient and Hessian
    const { gradient, hessian } = computeGradientHessian(Xmat, y, coefficients, K);

    // Newton-Raphson step: β_new = β_old - H^{-1} * g
    // Since we want to maximize, and H is negative definite, we solve: H * δ = g
    try {
      const H = new Matrix(hessian);
      const g = Matrix.columnVector(gradient);

      // Solve H * delta = -g (Newton direction)
      const delta = H.solve(g.mul(-1));

      // Update coefficients
      for (let k = 0; k < K - 1; k++) {
        for (let j = 0; j < p; j++) {
          const idx = k * p + j;
          coefficients[k][j] += delta.get(idx, 0);
        }
      }
    } catch (e) {
      // If Hessian is singular, try gradient ascent with small step
      console.warn(`Iteration ${iteration}: Hessian singular, using gradient ascent`);
      const stepSize = 0.01;
      for (let k = 0; k < K - 1; k++) {
        for (let j = 0; j < p; j++) {
          const idx = k * p + j;
          coefficients[k][j] += stepSize * gradient[idx];
        }
      }
    }

    // Compute log-likelihood
    logLik = computeLogLikelihood(Xmat, y, coefficients, K);

    // Check convergence
    if (iteration > 0 && Math.abs(logLik - logLikPrev) < tol) {
      converged = true;
      break;
    }
  }

  // Compute standard errors from Hessian
  const { hessian: finalHessian } = computeGradientHessian(Xmat, y, coefficients, K);
  let standardErrors = Array(K - 1).fill(null).map(() => Array(p).fill(NaN));

  try {
    const H = new Matrix(finalHessian);
    const covMatrix = H.inverse().mul(-1); // Negative inverse of Hessian

    for (let k = 0; k < K - 1; k++) {
      for (let j = 0; j < p; j++) {
        const idx = k * p + j;
        standardErrors[k][j] = Math.sqrt(Math.max(0, covMatrix.get(idx, idx)));
      }
    }
  } catch (e) {
    console.warn('Could not compute standard errors (Hessian not invertible)');
  }

  // Compute fitted probabilities
  const fitted = Array(n).fill(null).map(() => Array(K).fill(0));
  for (let i = 0; i < n; i++) {
    const logits = new Array(K);
    logits[0] = 0;

    for (let k = 1; k < K; k++) {
      let logit = 0;
      for (let j = 0; j < p; j++) {
        logit += coefficients[k - 1][j] * Xmat[i][j];
      }
      logits[k] = logit;
    }

    fitted[i] = softmax(logits);
  }

  // Compute deviance
  const deviance = -2 * logLik;

  // Null model (intercept only)
  const classCounts = new Array(K).fill(0);
  for (let i = 0; i < n; i++) {
    classCounts[y[i]]++;
  }
  const nullProbs = classCounts.map((c) => c / n);
  let nullLogLik = 0;
  for (let i = 0; i < n; i++) {
    nullLogLik += Math.log(Math.max(nullProbs[y[i]], 1e-15));
  }
  const nullDeviance = -2 * nullLogLik;

  // Compute AIC and BIC
  const nParams = (K - 1) * p;
  const aic = -2 * logLik + 2 * nParams;
  const bic = -2 * logLik + Math.log(n) * nParams;

  // Pseudo R²
  const pseudoR2 = 1 - deviance / nullDeviance;

  return {
    coefficients,
    standardErrors,
    fitted,
    logLikelihood: logLik,
    deviance,
    nullDeviance,
    pseudoR2,
    aic,
    bic,
    iterations: converged ? iteration + 1 : iteration,
    converged,
    n,
    p,
    K,
    nParams,
    intercept,
  };
}

/**
 * Predict from multinomial logistic regression
 * @param {Object} model - Fitted model from fitMultinomial
 * @param {Array<Array<number>>} X - New data
 * @param {Object} options - Prediction options
 * @returns {Array} Predictions
 */
export function predictMultinomial(model, X, options = {}) {
  const { type = 'class' } = options;

  const n = X.length;
  const K = model.K;
  const p = model.p;

  // Add intercept if needed
  let Xmat = model.intercept ? X.map((row) => [1, ...row]) : X.map((row) => [...row]);

  // Compute probabilities
  const probabilities = Array(n).fill(null).map(() => Array(K).fill(0));

  for (let i = 0; i < n; i++) {
    const logits = new Array(K);
    logits[0] = 0; // reference class

    for (let k = 1; k < K; k++) {
      let logit = 0;
      for (let j = 0; j < p; j++) {
        logit += model.coefficients[k - 1][j] * Xmat[i][j];
      }
      logits[k] = logit;
    }

    probabilities[i] = softmax(logits);
  }

  if (type === 'probs' || type === 'proba') {
    return probabilities;
  }

  // Return class predictions
  const predictions = probabilities.map((probs) => {
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let k = 1; k < K; k++) {
      if (probs[k] > maxProb) {
        maxProb = probs[k];
        maxIdx = k;
      }
    }
    return maxIdx;
  });

  return predictions;
}
