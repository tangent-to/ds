/**
 * Distance metrics for ML algorithms
 * Centralized distance functions used by KNN, DBSCAN, clustering, etc.
 */

/**
 * Euclidean distance (L2 norm)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Euclidean distance
 */
export function euclidean(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Manhattan distance (L1 norm, taxicab distance)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Manhattan distance
 */
export function manhattan(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

/**
 * Minkowski distance (generalized Lp norm)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @param {number} p - Order parameter (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
 * @returns {number} Minkowski distance
 */
export function minkowski(a, b, p = 2) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  if (p === Infinity) {
    return chebyshev(a, b);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(Math.abs(a[i] - b[i]), p);
  }
  return Math.pow(sum, 1 / p);
}

/**
 * Chebyshev distance (L∞ norm, maximum metric)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Chebyshev distance
 */
export function chebyshev(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) {
      max = diff;
    }
  }
  return max;
}

/**
 * Cosine distance (1 - cosine similarity)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Cosine distance (0 = identical direction, 2 = opposite)
 */
export function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 1; // Orthogonal
  }

  const cosineSimilarity = dotProduct / (normA * normB);
  return 1 - cosineSimilarity;
}

/**
 * Hamming distance (for categorical/binary data)
 * Counts the number of positions where elements differ
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {number} Hamming distance
 */
export function hamming(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let count = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      count++;
    }
  }
  return count;
}

/**
 * Canberra distance (weighted version of Manhattan distance)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Canberra distance
 */
export function canberra(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const absSum = Math.abs(a[i]) + Math.abs(b[i]);
    if (absSum > 0) {
      sum += Math.abs(a[i] - b[i]) / absSum;
    }
  }
  return sum;
}

/**
 * Get distance function by name
 * @param {string|Function} metric - Metric name or custom function
 * @returns {Function} Distance function
 */
export function getDistanceFunction(metric) {
  if (typeof metric === 'function') {
    return metric;
  }

  const metrics = {
    euclidean,
    manhattan,
    minkowski,
    chebyshev,
    cosine,
    hamming,
    canberra
  };

  if (!metrics[metric]) {
    throw new Error(`Unknown distance metric: ${metric}. Available: ${Object.keys(metrics).join(', ')}`);
  }

  return metrics[metric];
}

// Default export
export default {
  euclidean,
  manhattan,
  minkowski,
  chebyshev,
  cosine,
  hamming,
  canberra,
  getDistanceFunction
};
