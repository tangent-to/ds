/**
 * Loss functions on plain arrays, each returning a number. For a loss to
 * train a network with, see `@tangent.to/nn`, whose losses are expressions
 * on the tape and differentiate themselves.
 */

/**
 * Mean Squared Error Loss
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number}
 */
export function mseLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const diff = yPred[i] - yTrue[i];
    loss += diff * diff;
  }
  loss /= n;
  
  return loss;
}

/**
 * Mean Absolute Error Loss
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number}
 */
export function maeLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += Math.abs(yPred[i] - yTrue[i]);
  }
  loss /= n;
  
  return loss;
}

/**
 * Binary Cross-Entropy Loss (Log Loss)
 * @param {Array<number>} yTrue - True labels (0 or 1)
 * @param {Array<number>} yPred - Predicted probabilities
 * @param {number} epsilon - Small value to avoid log(0)
 * @returns {number}
 */
export function logLoss(yTrue, yPred, epsilon = 1e-15) {
  const n = yTrue.length;
  
  // Clip predictions to avoid log(0)
  const yPredClipped = yPred.map(p => Math.max(epsilon, Math.min(1 - epsilon, p)));
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += -(yTrue[i] * Math.log(yPredClipped[i]) + 
              (1 - yTrue[i]) * Math.log(1 - yPredClipped[i]));
  }
  loss /= n;
  
  return loss;
}

/**
 * Categorical Cross-Entropy Loss
 * @param {Array<Array<number>>} yTrue - One-hot encoded true labels
 * @param {Array<Array<number>>} yPred - Predicted probabilities
 * @param {number} epsilon - Small value to avoid log(0)
 * @returns {number}
 */
export function crossEntropy(yTrue, yPred, epsilon = 1e-15) {
  const n = yTrue.length;
  const k = yTrue[0].length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const pred = Math.max(epsilon, Math.min(1 - epsilon, yPred[i][j]));
      loss += -yTrue[i][j] * Math.log(pred);
    }
  }
  loss /= n;
  
  return loss;
}

/**
 * Hinge Loss (for SVM)
 * @param {Array<number>} yTrue - True labels (-1 or 1)
 * @param {Array<number>} yPred - Predicted scores
 * @returns {number}
 */
export function hingeLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += Math.max(0, 1 - yTrue[i] * yPred[i]);
  }
  loss /= n;
  
  return loss;
}

/**
 * Huber Loss (robust to outliers)
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @param {number} delta - Threshold for switching from quadratic to linear
 * @returns {number}
 */
export function huberLoss(yTrue, yPred, delta = 1.0) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const diff = Math.abs(yPred[i] - yTrue[i]);
    if (diff <= delta) {
      loss += 0.5 * diff * diff;
    } else {
      loss += delta * (diff - 0.5 * delta);
    }
  }
  loss /= n;
  
  return loss;
}

/**
 * Get loss function by name
 * @param {string} name - Loss function name
 * @returns {Function} Loss function
 */
export function getLossFunction(name) {
  const losses = {
    'mse': mseLoss,
    'mean_squared_error': mseLoss,
    'mae': maeLoss,
    'mean_absolute_error': maeLoss,
    'log': logLoss,
    'log_loss': logLoss,
    'binary_crossentropy': logLoss,
    'crossentropy': crossEntropy,
    'categorical_crossentropy': crossEntropy,
    'hinge': hingeLoss,
    'huber': huberLoss
  };

  const lossFn = losses[name.toLowerCase()];
  if (!lossFn) {
    throw new Error(`Unknown loss function: ${name}. Available: ${Object.keys(losses).join(', ')}`);
  }

  return lossFn;
}
