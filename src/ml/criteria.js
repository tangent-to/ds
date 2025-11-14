/**
 * Impurity/splitting criteria for tree-based algorithms
 * Centralized impurity functions used by DecisionTree, RandomForest, etc.
 */

/**
 * Gini impurity for classification
 * Measures probability of misclassification
 * Lower is better (0 = pure node)
 * @param {Array} labels - Array of labels
 * @returns {number} Gini impurity [0, 1]
 */
export function gini(labels) {
  if (labels.length === 0) return 0;

  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });

  const total = labels.length;
  let sum = 0;

  for (const count of counts.values()) {
    const p = count / total;
    sum += p * p;
  }

  return 1 - sum;
}

/**
 * Entropy (information gain) for classification
 * Measures uncertainty/disorder in the data
 * Lower is better (0 = pure node)
 * @param {Array} labels - Array of labels
 * @returns {number} Entropy [0, log2(n_classes)]
 */
export function entropy(labels) {
  if (labels.length === 0) return 0;

  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });

  const total = labels.length;
  let sum = 0;

  for (const count of counts.values()) {
    const p = count / total;
    if (p > 0) {
      sum -= p * Math.log2(p);
    }
  }

  return sum;
}

/**
 * Variance for regression
 * Measures spread of continuous values
 * Lower is better (0 = all values equal)
 * @param {Array<number>} values - Array of numeric values
 * @returns {number} Variance
 */
export function variance(values) {
  if (values.length === 0) return 0;

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  let sum = 0;

  for (const v of values) {
    const diff = v - mean;
    sum += diff * diff;
  }

  return sum / values.length;
}

/**
 * Mean Squared Error (MSE) for regression
 * Alternative to variance, measures prediction error
 * @param {Array<number>} values - Array of numeric values
 * @returns {number} MSE (same as variance for single node)
 */
export function mse(values) {
  return variance(values);
}

/**
 * Mean Absolute Error (MAE) for regression
 * Robust alternative to MSE
 * @param {Array<number>} values - Array of numeric values
 * @returns {number} MAE
 */
export function mae(values) {
  if (values.length === 0) return 0;

  const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  let sum = 0;

  for (const v of values) {
    sum += Math.abs(v - median);
  }

  return sum / values.length;
}

/**
 * Classification error (misclassification rate)
 * Simple impurity measure based on majority class
 * @param {Array} labels - Array of labels
 * @returns {number} Classification error [0, 1]
 */
export function classificationError(labels) {
  if (labels.length === 0) return 0;

  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });

  const maxCount = Math.max(...counts.values());
  return 1 - (maxCount / labels.length);
}

/**
 * Get impurity function by name
 * @param {string|Function} criterion - Criterion name or custom function
 * @param {string} task - 'classification' or 'regression'
 * @returns {Function} Impurity function
 */
export function getCriterionFunction(criterion, task = 'classification') {
  if (typeof criterion === 'function') {
    return criterion;
  }

  const classificationCriteria = {
    gini,
    entropy,
    classification_error: classificationError
  };

  const regressionCriteria = {
    variance,
    mse,
    mae
  };

  const criteria = task === 'classification' ? classificationCriteria : regressionCriteria;

  if (!criteria[criterion]) {
    const available = Object.keys(criteria).join(', ');
    throw new Error(`Unknown criterion: ${criterion} for task ${task}. Available: ${available}`);
  }

  return criteria[criterion];
}

/**
 * Compute information gain (reduction in impurity)
 * @param {Array} parentLabels - Labels before split
 * @param {Array} leftLabels - Labels in left child
 * @param {Array} rightLabels - Labels in right child
 * @param {Function} impurityFn - Impurity function (gini, entropy, etc.)
 * @returns {number} Information gain
 */
export function informationGain(parentLabels, leftLabels, rightLabels, impurityFn = gini) {
  const parentImpurity = impurityFn(parentLabels);
  const n = parentLabels.length;
  const nLeft = leftLabels.length;
  const nRight = rightLabels.length;

  if (nLeft === 0 || nRight === 0) {
    return 0;
  }

  const leftImpurity = impurityFn(leftLabels);
  const rightImpurity = impurityFn(rightLabels);

  const weightedImpurity = (nLeft / n) * leftImpurity + (nRight / n) * rightImpurity;

  return parentImpurity - weightedImpurity;
}

// Default export
export default {
  gini,
  entropy,
  variance,
  mse,
  mae,
  classificationError,
  getCriterionFunction,
  informationGain
};
