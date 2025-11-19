/**
 * Enhanced HDBSCAN estimator with performance optimizations
 *
 * New features:
 * - Custom distance metrics (euclidean, manhattan, cosine, etc.)
 * - Fast mode using KD-tree (O(n log n) vs O(n²))
 * - Improved cluster stability calculation
 * - Incremental learning support
 *
 * Usage:
 *   const hdbscan = new HDBSCANFast({
 *     minClusterSize: 5,
 *     metric: 'manhattan',
 *     algorithm: 'kdtree'
 *   });
 *   hdbscan.fit(data);
 *
 *   // Incremental update
 *   hdbscan.partialFit(newBatch);
 */

import { Estimator } from '../../core/estimators/estimator.js';
import { fitFast } from '../hdbscan_fast.js';
import * as hdbscanFn from '../hdbscan.js';
import { prepareX } from '../../core/table.js';
import { getDistanceFunction } from '../distances.js';

export class HDBSCANFast extends Estimator {
  /**
   * @param {Object} params - Configuration
   * @param {number} [params.minClusterSize=5] - Minimum cluster size
   * @param {number} [params.minSamples=null] - Minimum samples (defaults to minClusterSize)
   * @param {string} [params.clusterSelectionMethod='eom'] - Cluster selection method
   * @param {string|Function} [params.metric='euclidean'] - Distance metric
   * @param {string} [params.algorithm='kdtree'] - Algorithm: 'kdtree' or 'standard'
   */
  constructor({
    minClusterSize = 5,
    minSamples = null,
    clusterSelectionMethod = 'eom',
    metric = 'euclidean',
    algorithm = 'kdtree'
  } = {}) {
    super({ minClusterSize, minSamples, clusterSelectionMethod, metric, algorithm });

    this.minClusterSize = minClusterSize;
    this.minSamples = minSamples;
    this.clusterSelectionMethod = clusterSelectionMethod;
    this.metric = metric;
    this.algorithm = algorithm;

    this.model = null;
    this.fitted = false;
    this.X_train = null;
    this.incrementalData = [];  // For incremental learning
  }

  /**
   * Fit the HDBSCAN model
   * @param {Array|Object} X - Data or options object
   * @param {Object} opts - Additional options
   * @returns {this}
   */
  fit(X, opts = {}) {
    let fitResult;
    let trainData;

    // Handle declarative input
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const callOpts = {
        minClusterSize: X.minClusterSize !== undefined ? X.minClusterSize : this.minClusterSize,
        minSamples: X.minSamples !== undefined ? X.minSamples : this.minSamples,
        clusterSelectionMethod: X.clusterSelectionMethod !== undefined ? X.clusterSelectionMethod : this.clusterSelectionMethod,
        metric: X.metric !== undefined ? X.metric : this.metric,
        algorithm: X.algorithm !== undefined ? X.algorithm : this.algorithm,
        columns: X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      };

      const prepared = prepareX({
        columns: callOpts.columns,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      trainData = prepared.X;

      // Use fast implementation
      fitResult = fitFast(callOpts);
    } else {
      // Numeric input
      const callOpts = {
        minClusterSize: opts.minClusterSize !== undefined ? opts.minClusterSize : this.minClusterSize,
        minSamples: opts.minSamples !== undefined ? opts.minSamples : this.minSamples,
        clusterSelectionMethod: opts.clusterSelectionMethod !== undefined ? opts.clusterSelectionMethod : this.clusterSelectionMethod,
        metric: opts.metric !== undefined ? opts.metric : this.metric,
        algorithm: opts.algorithm !== undefined ? opts.algorithm : this.algorithm
      };

      fitResult = fitFast(X, callOpts);
      trainData = Array.isArray(X[0]) ? X : X.map((x) => [x]);
    }

    // Store model
    this.model = fitResult;
    this.labels = fitResult.labels;
    this.probabilities = fitResult.probabilities;
    this.nClusters = fitResult.nClusters;
    this.nNoise = fitResult.nNoise;
    this.hierarchy = fitResult.hierarchy;
    this.condensedTree = fitResult.condensedTree;
    this.stabilities = fitResult.stabilities;
    this.coreDistances = fitResult.coreDistances;
    this.X_train = trainData;
    this.incrementalData = [trainData];
    this.fitted = true;

    return this;
  }

  /**
   * Partial fit for incremental learning
   * Combines new data with existing data and refits
   *
   * @param {Array|Object} X - New data batch
   * @param {Object} opts - Options
   * @returns {this}
   */
  partialFit(X, opts = {}) {
    if (!this.fitted) {
      // First batch - just fit
      return this.fit(X, opts);
    }

    // Prepare new data
    let newData;
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      newData = prepared.X;
    } else {
      newData = Array.isArray(X[0]) ? X : X.map((x) => [x]);
    }

    // Combine with existing data
    this.incrementalData.push(newData);
    const combinedData = this.incrementalData.flat();

    // Refit on combined data
    const fitResult = fitFast(combinedData, {
      minClusterSize: this.minClusterSize,
      minSamples: this.minSamples,
      metric: this.metric,
      algorithm: this.algorithm,
      clusterSelectionMethod: this.clusterSelectionMethod
    });

    // Update model
    this.model = fitResult;
    this.labels = fitResult.labels;
    this.probabilities = fitResult.probabilities;
    this.nClusters = fitResult.nClusters;
    this.nNoise = fitResult.nNoise;
    this.hierarchy = fitResult.hierarchy;
    this.condensedTree = fitResult.condensedTree;
    this.stabilities = fitResult.stabilities;
    this.coreDistances = fitResult.coreDistances;
    this.X_train = combinedData;

    return this;
  }

  /**
   * Predict cluster labels for new data
   * @param {Array|Object} X - New data
   * @returns {Object} {labels, probabilities}
   */
  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('HDBSCANFast: estimator is not fitted. Call fit() first.');
    }

    // Prepare data
    let testData;
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      testData = prepared.X;
    } else {
      testData = Array.isArray(X[0]) ? X : X.map((x) => [x]);
    }

    // Use nearest neighbor assignment
    return hdbscanFn.predict(this.model, testData, this.X_train);
  }

  /**
   * Get cluster persistence
   * @returns {Array<Object>}
   */
  getClusterPersistence() {
    if (!this.fitted) {
      throw new Error('HDBSCANFast: estimator is not fitted.');
    }
    return hdbscanFn.clusterPersistence(this.model);
  }

  /**
   * Get condensed tree
   * @returns {Array<Object>}
   */
  getCondensedTree() {
    if (!this.fitted) {
      throw new Error('HDBSCANFast: estimator is not fitted.');
    }
    return this.condensedTree;
  }

  /**
   * Get hierarchy
   * @returns {Object}
   */
  getHierarchy() {
    if (!this.fitted) {
      throw new Error('HDBSCANFast: estimator is not fitted.');
    }
    return this.hierarchy;
  }

  /**
   * Get cluster stability scores
   * @returns {Array<Object>}
   */
  getStabilities() {
    if (!this.fitted) {
      throw new Error('HDBSCANFast: estimator is not fitted.');
    }
    return this.stabilities;
  }

  /**
   * Summary statistics
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('HDBSCANFast: estimator is not fitted.');
    }

    const nSamples = this.labels.length;
    const avgProbability = this.probabilities.reduce((a, b) => a + b, 0) / nSamples;

    const clusterCounts = {};
    for (const label of this.labels) {
      if (label !== -1) {
        clusterCounts[label] = (clusterCounts[label] || 0) + 1;
      }
    }

    // Stability summary
    const stabilityScores = this.stabilities.map(s => s.stability);
    const avgStability = stabilityScores.length > 0
      ? stabilityScores.reduce((a, b) => a + b, 0) / stabilityScores.length
      : 0;

    return {
      minClusterSize: this.minClusterSize,
      minSamples: this.minSamples || this.minClusterSize,
      metric: this.metric,
      algorithm: this.algorithm,
      clusterSelectionMethod: this.clusterSelectionMethod,
      nClusters: this.nClusters,
      nNoise: this.nNoise,
      nSamples,
      noiseRatio: this.nNoise / nSamples,
      avgProbability,
      avgStability,
      clusterSizes: clusterCounts,
      stabilities: this.stabilities
    };
  }

  /**
   * Serialization
   */
  toJSON() {
    return {
      __class__: 'HDBSCANFast',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
      X_train: this.X_train,
      incrementalData: this.incrementalData
    };
  }

  static fromJSON(obj = {}) {
    const inst = new HDBSCANFast(obj.params || {});
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
      inst.incrementalData = obj.incrementalData || [obj.X_train];
      inst.fitted = true;
    }
    return inst;
  }
}

export default HDBSCANFast;
