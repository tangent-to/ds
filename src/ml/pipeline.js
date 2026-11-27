/**
 * Pipeline utilities for chaining preprocessing and models
 */

import { Estimator } from '../core/estimators/estimator.js';
import { prepareX, prepareXY } from '../core/table.js';
import { GLM } from '../stats/estimators/GLM.js';
import { MLPRegressor } from './estimators/MLPRegressor.js';
import { StandardScaler } from './preprocessing.js';
import { kFold } from './validation.js';

const serializableSteps = new Map(Object.entries({ GLM, MLPRegressor, StandardScaler }));

function isTableDescriptor(input) {
  return (
    input &&
    typeof input === 'object' &&
    !Array.isArray(input) &&
    (input.data || input.rows)
  );
}

/**
 * Pipeline class for chaining transformers and estimators
 */
export class Pipeline extends Estimator {
  /**
   * Create a pipeline
   * @param {Array<Object>} steps - Array of transformers/estimators
   * @param {Object} [options]
   * @param {string[]} [options.columns] - Ordered numeric features for named input rows
   */
  constructor(steps, { columns = null } = {}) {
    if (!Array.isArray(steps) || steps.length === 0) {
      throw new Error('Pipeline requires at least one step');
    }
    if (columns !== null && (!Array.isArray(columns) || columns.length === 0 ||
      Array.from(columns).some((column) => typeof column !== 'string' || column.length === 0) ||
      new Set(columns).size !== columns.length)) {
      throw new Error('Pipeline columns must be a non-empty array of unique feature names');
    }

    columns = columns?.slice() ?? null;
    super(columns === null ? { steps } : { steps, columns });
    this.steps = steps;
    this.columns = columns;
  }

  _prepareFeatures(X) {
    // Keep rows aligned with positional targets and predictions. A missing
    // numeric feature raises an error instead of silently dropping its row.
    return this.columns === null ? X : prepareX({ data: X, columns: this.columns, naOmit: false }).X;
  }

  /**
   * Fit all steps
   * @param {Array} X - Feature matrix, or named rows when columns are configured
   * @param {Array} y - Target values (optional)
   * @returns {Pipeline} this
   */
  fit(X, y = null) {
    let XTransformed = this._prepareFeatures(X);
    this.fitted = false;

    // Fit and transform through all but last step
    for (let i = 0; i < this.steps.length - 1; i++) {
      const step = this.steps[i];

      if (typeof step.fit !== 'function') {
        throw new Error(`Step ${i} must have a fit() method`);
      }

      step.fit(XTransformed, y);

      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    // Fit final step
    const finalStep = this.steps[this.steps.length - 1];
    if (typeof finalStep.fit === 'function') {
      finalStep.fit(XTransformed, y);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform data through all steps
   * @param {Array} X - Feature matrix, or named rows when columns are configured
   * @returns {Array} Transformed data
   */
  transform(X) {
    this._ensureFitted('transform');

    let XTransformed = this._prepareFeatures(X);

    for (const step of this.steps) {
      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    return XTransformed;
  }

  /**
   * Fit and transform in one step
   * @param {Array} X - Feature matrix, or named rows when columns are configured
   * @param {Array} y - Target values (optional)
   * @returns {Array} Transformed data
   */
  fitTransform(X, y = null) {
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Predict using final estimator
   * @param {Array} X - Feature matrix
   * @returns {Array} Predictions
   */
  predict(X) {
    this._ensureFitted('predict');

    let XTransformed = this._prepareFeatures(X);

    // Transform through all but last step
    for (let i = 0; i < this.steps.length - 1; i++) {
      const step = this.steps[i];
      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    // Predict with final step
    const finalStep = this.steps[this.steps.length - 1];
    if (typeof finalStep.predict !== 'function') {
      throw new Error('Final step must have a predict() method');
    }

    return finalStep.predict(XTransformed);
  }

  /**
   * Get final estimator
   * @returns {Object} Final step
   */
  getFinalEstimator() {
    return this.steps[this.steps.length - 1];
  }

  /**
   * Export a fitted pipeline with the feature order needed at prediction time.
   * Supports StandardScaler, GLM and MLPRegressor steps.
   */
  toJSON() {
    this._ensureFitted('toJSON');
    return {
      format: 'ds/pipeline/1',
      columns: this.columns?.slice() ?? null,
      steps: this.steps.map((step, i) => {
        const Step = step?.constructor;
        if (![...serializableSteps.values()].includes(Step)) {
          throw new Error(`Pipeline cannot serialize step ${i} (${Step?.name})`);
        }
        return step.toJSON();
      }),
    };
  }

  /** Restore a fitted pipeline from its JSON representation. */
  static fromJSON(json) {
    if (json?.format !== 'ds/pipeline/1' || !Array.isArray(json.steps)) {
      throw new Error('Invalid pipeline format');
    }
    const steps = json.steps.map((state, i) => {
      const Step = serializableSteps.get(state?.__class__);
      if (!Step) throw new Error(`Pipeline cannot restore step ${i} (${state?.__class__})`);
      const step = Step.fromJSON(state);
      if (Step !== StandardScaler && !step.isFitted()) {
        throw new Error(`Pipeline cannot restore unfitted step ${i}`);
      }
      return step;
    });
    const pipeline = new Pipeline(steps, { columns: json.columns });
    pipeline.fitted = true;
    return pipeline;
  }
}

/**
 * Simple GridSearchCV for hyperparameter tuning
 */
export class GridSearchCV extends Estimator {
  /**
   * Create grid search
   * @param {Function} estimatorFn - Function that creates estimator: (params) => estimator
   * @param {Object} paramGrid - Grid of parameters: {param1: [val1, val2], ...}
   * @param {Function} scoreFn - Scoring function: (yTrue, yPred) => score
   * @param {number} cv - Number of cross-validation folds
   */
  constructor(estimatorFn, paramGrid, scoreFn, cv = 5) {
    super({ cv });
    this.estimatorFn = estimatorFn;
    this.paramGrid = paramGrid;
    this.scoreFn = scoreFn;
    this.cv = cv;
    this.bestParams = null;
    this.bestScore = -Infinity;
    this.bestEstimator = null;
    this.cvResults = [];
  }

  /**
   * Generate all parameter combinations
   * @private
   */
  _generateParamCombinations() {
    const keys = Object.keys(this.paramGrid);
    const values = keys.map((k) => this.paramGrid[k]);

    const combinations = [];

    const generate = (current, depth) => {
      if (depth === keys.length) {
        combinations.push({ ...current });
        return;
      }

      for (const value of values[depth]) {
        current[keys[depth]] = value;
        generate(current, depth + 1);
      }
    };

    generate({}, 0);
    return combinations;
  }

  /**
   * Perform grid search with cross-validation
   * @param {Array} X - Feature matrix
   * @param {Array} y - Target values
   * @returns {GridSearchCV} this
   */
  fit(X, y) {
    let dataX = X;
    let dataY = y;

    if (isTableDescriptor(X)) {
      if (!X.y) {
        throw new Error('GridSearchCV.fit requires a y column when using table descriptors');
      }
      const prepared = prepareXY({
        X: X.X || X.columns,
        y: X.y,
        data: X.data || X.rows,
        naOmit: X.naOmit,
        omit_missing: X.omit_missing,
        encode: X.encode,
      });
      dataX = prepared.X;
      dataY = prepared.y;
    }

    if (!Array.isArray(dataX) || !Array.isArray(dataY)) {
      throw new Error('GridSearchCV.fit expects array inputs for X and y');
    }

    this.bestParams = null;
    this.bestScore = -Infinity;
    this.bestEstimator = null;
    this.cvResults = [];

    const combinations = this._generateParamCombinations();
    const folds = Array.isArray(this.cv)
      ? this.cv
      : kFold(dataX, dataY, this.cv, false);

    for (const params of combinations) {
      const scores = [];

      // Cross-validation on the shared balanced folds
      for (const fold of folds) {
        const XTrain = fold.trainIndices.map((i) => dataX[i]);
        const yTrain = fold.trainIndices.map((i) => dataY[i]);
        const XTest = fold.testIndices.map((i) => dataX[i]);
        const yTest = fold.testIndices.map((i) => dataY[i]);

        const estimator = this.estimatorFn(params);
        estimator.fit(XTrain, yTrain);
        const yPred = estimator.predict(XTest);
        const score = this.scoreFn(yTest, yPred);
        scores.push(score);
      }

      const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;

      this.cvResults.push({
        params,
        meanScore,
        scores,
      });

      if (meanScore > this.bestScore) {
        this.bestScore = meanScore;
        this.bestParams = params;
      }
    }

    // Fit best estimator on full data
    this.bestEstimator = this.estimatorFn(this.bestParams);
    this.bestEstimator.fit(dataX, dataY);

    this.fitted = true;
    return this;
  }

  /**
   * Predict using best estimator
   * @param {Array} X - Feature matrix
   * @returns {Array} Predictions
   */
  predict(X) {
    this._ensureFitted('predict');
    return this.bestEstimator.predict(X);
  }

  /**
   * Get results sorted by score
   * @returns {Array<Object>} Results
   */
  getResults() {
    return [...this.cvResults].sort((a, b) => b.meanScore - a.meanScore);
  }
}
