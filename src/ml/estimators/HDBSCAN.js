/**
 * HDBSCAN estimator - class wrapper around functional hdbscan utilities
 *
 * Usage:
 *   const hdbscan = new HDBSCAN({ minClusterSize: 5, minSamples: 5 });
 *   hdbscan.fit({ data: myData, columns: ['x', 'y'] });
 *   const labels = hdbscan.labels;  // -1 for noise, 0+ for cluster IDs
 *   const probabilities = hdbscan.probabilities; // Cluster membership probabilities
 *
 * The class accepts both numeric-array inputs and the declarative table-style objects
 * supported by the core table utilities.
 */

import { Estimator } from '../../core/estimators/estimator.js';
import * as hdbscanFn from '../hdbscan.js';
import { prepareX } from '../../core/table.js';

export class HDBSCAN extends Estimator {
  /**
   * @param {Object} params - Configuration parameters
   * @param {number} [params.minClusterSize=5] - Minimum cluster size
   * @param {number} [params.minSamples=null] - Minimum samples for core distance (defaults to minClusterSize)
   * @param {string} [params.clusterSelectionMethod='eom'] - 'eom' (Excess of Mass) or 'leaf'
   */
  constructor({
    minClusterSize = 5,
    minSamples = null,
    clusterSelectionMethod = 'eom'
  } = {}) {
    super({ minClusterSize, minSamples, clusterSelectionMethod });
    this.minClusterSize = minClusterSize;
    this.minSamples = minSamples;
    this.clusterSelectionMethod = clusterSelectionMethod;

    // fitted model placeholder (result of hdbscanFn.fit)
    this.model = null;
    this.fitted = false;
    this.X_train = null;  // Store training data for prediction
  }

  /**
   * Fit the HDBSCAN model.
   *
   * Accepts:
   *  - numeric input: fit(Xarray, { minClusterSize, minSamples })
   *  - declarative input: fit({ data: tableLike, columns: ['c1','c2'], minClusterSize, ... })
   *
   * Returns this.
   */
  fit(X, opts = {}) {
    // If invoked with a single options-object that contains `data` or `columns`,
    // forward as declarative call to the underlying function.
    let fitResult;
    let trainData;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns)
    ) {
      const callOpts = {
        minClusterSize: X.minClusterSize !== undefined ? X.minClusterSize : this.minClusterSize,
        minSamples: X.minSamples !== undefined ? X.minSamples : this.minSamples,
        clusterSelectionMethod: X.clusterSelectionMethod !== undefined ? X.clusterSelectionMethod : this.clusterSelectionMethod,
        columns: X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      };

      // Prepare data to store for prediction
      const prepared = prepareX({
        columns: callOpts.columns,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      trainData = prepared.X;

      // underlying hdbscan.fit supports declarative object form
      fitResult = hdbscanFn.fit(callOpts);
    } else {
      // Positional numeric-style call
      const callOpts = {
        minClusterSize: opts.minClusterSize !== undefined ? opts.minClusterSize : this.minClusterSize,
        minSamples: opts.minSamples !== undefined ? opts.minSamples : this.minSamples,
        clusterSelectionMethod: opts.clusterSelectionMethod !== undefined ? opts.clusterSelectionMethod : this.clusterSelectionMethod
      };
      fitResult = hdbscanFn.fit(X, callOpts);
      trainData = Array.isArray(X[0]) ? X : X.map((x) => [x]);
    }

    // store model details
    this.model = fitResult;
    this.labels = fitResult.labels;
    this.labels_ = fitResult.labels; // Alias for sklearn compatibility
    this.probabilities = fitResult.probabilities;
    this.probabilities_ = fitResult.probabilities; // Alias for sklearn compatibility
    this.nClusters = fitResult.nClusters;
    this.nNoise = fitResult.nNoise;
    this.hierarchy = fitResult.hierarchy;
    this.condensedTree = fitResult.condensedTree;
    this.stabilities = fitResult.stabilities;
    this.coreDistances = fitResult.coreDistances;
    this.X_train = trainData;
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for new data.
   *
   * Note: HDBSCAN uses approximate nearest neighbor assignment for prediction.
   * New points are assigned to the cluster of their nearest neighbor if within
   * the core distance threshold.
   *
   * Accepts:
   *  - numeric array: predict([[x1,x2], [x1,x2], ...])
   *  - declarative: predict({ data: tableLike, columns: ['c1','c2'], omit_missing: true })
   *
   * @returns {Object} {labels, probabilities}
   */
  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('HDBSCAN: estimator is not fitted. Call fit() first.');
    }

    // If declarative object with data/columns provided, prepare numeric matrix
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      return hdbscanFn.predict(this.model, prepared.X, this.X_train);
    }

    // Otherwise assume numeric arrays and delegate to functional predict
    return hdbscanFn.predict(this.model, X, this.X_train);
  }

  /**
   * Get cluster persistence scores
   * @returns {Array<Object>} Array of {cluster, persistence} objects
   */
  getClusterPersistence() {
    if (!this.fitted) {
      throw new Error('HDBSCAN: estimator is not fitted.');
    }
    return hdbscanFn.clusterPersistence(this.model);
  }

  /**
   * Get the condensed cluster tree for visualization
   * @returns {Array<Object>} Condensed tree structure
   */
  getCondensedTree() {
    if (!this.fitted) {
      throw new Error('HDBSCAN: estimator is not fitted.');
    }
    return this.condensedTree;
  }

  /**
   * Get the full hierarchy (dendrogram)
   * @returns {Object} Hierarchy structure with dendrogram and linkage matrix
   */
  getHierarchy() {
    if (!this.fitted) {
      throw new Error('HDBSCAN: estimator is not fitted.');
    }
    return this.hierarchy;
  }

  /**
   * Convenience: return summary stats for fitted model
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('HDBSCAN: estimator is not fitted.');
    }

    const nSamples = this.labels.length;
    const avgProbability = this.probabilities.reduce((a, b) => a + b, 0) / nSamples;

    // Count samples per cluster
    const clusterCounts = {};
    for (const label of this.labels) {
      if (label !== -1) {
        clusterCounts[label] = (clusterCounts[label] || 0) + 1;
      }
    }

    return {
      minClusterSize: this.minClusterSize,
      minSamples: this.minSamples || this.minClusterSize,
      clusterSelectionMethod: this.clusterSelectionMethod,
      nClusters: this.nClusters,
      nNoise: this.nNoise,
      nSamples,
      noiseRatio: this.nNoise / nSamples,
      avgProbability,
      clusterSizes: clusterCounts
    };
  }

  /**
   * Serialization helper
   */
  toJSON() {
    return {
      __class__: 'HDBSCAN',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
      X_train: this.X_train
    };
  }

  static fromJSON(obj = {}) {
    const inst = new HDBSCAN(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.labels = obj.model.labels;
      inst.probabilities = obj.model.probabilities;
      inst.nClusters = obj.model.nClusters;
      inst.nNoise = obj.model.nNoise;
      inst.hierarchy = obj.model.hierarchy;
      inst.condensedTree = obj.model.condensedTree;
      inst.stabilities = obj.model.stabilities;
      inst.coreDistances = obj.model.coreDistances;
      inst.X_train = obj.X_train;
      inst.fitted = true;
    }
    return inst;
  }
}

export default HDBSCAN;
