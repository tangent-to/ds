/**
 * MLPRegressor: a fully connected network as a ds estimator, built on
 * `@tangent.to/nn`. The chain `layerSizes` describes, dropout between the
 * hidden layers if asked, mean squared error, targets standardized by
 * default. Synchronous, so it sits in a `Pipeline` like any other estimator;
 * for branches, other losses, uncertainty and the asynchronous loop, use nn
 * directly.
 */

import nn from '@tangent.to/nn';
import { Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';

const DEFAULT_PARAMS = {
  layerSizes: null,
  activation: 'relu',
  optimizer: 'adam',
  learningRate: 0.01,
  epochs: 100,
  batchSize: 32,
  dropout: 0,
  normalizeY: true,
  seed: null,
  verbose: false,
  omit_missing: true,
};

const isSpec = (X) => X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.X || X.columns);

export class MLPRegressor extends Regressor {
  /**
   * @param {Object} [params]
   * @param {number[]|null} [params.layerSizes] - `[inputs, ...hidden, outputs]`;
   *   by default one hidden layer of `max(4, 2 · inputs)` units and one output
   * @param {string} [params.activation='relu'] - of the hidden layers
   * @param {'adam'|'sgd'|'momentum'|'rmsprop'|'lbfgs'} [params.optimizer='adam']
   * @param {number} [params.learningRate=0.01]
   * @param {number} [params.epochs=100] - iterations, for L-BFGS
   * @param {number} [params.batchSize=32]
   * @param {number} [params.dropout=0] - rate, applied after each hidden layer
   * @param {boolean} [params.normalizeY=true] - standardize the targets for training
   * @param {number|null} [params.seed] - for initialization, shuffles and masks
   * @param {boolean} [params.verbose=false]
   */
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
    this.history = null;
  }

  /**
   * Fit on `(X, y)`, or on a declarative spec `{ X, columns, y, data, omit_missing }`.
   * @returns {this}
   */
  fit(X, y = null, opts = {}) {
    let dataX = X;
    let dataY = y;
    const merged = { ...this.params, ...opts };
    if (isSpec(X)) {
      const callOpts = { ...merged, ...X };
      const prepared = prepareXY({ X: callOpts.X || callOpts.columns, y: callOpts.y, data: callOpts.data, omit_missing: callOpts.omit_missing });
      dataX = prepared.X;
      dataY = prepared.y;
      for (const k of Object.keys(DEFAULT_PARAMS)) if (k in X) merged[k] = X[k];
    }
    if (!dataX || !dataY) throw new Error('MLPRegressor.fit requires X and y.');

    const d = dataX[0].length;
    const outputs = Array.isArray(dataY[0]) ? dataY[0].length : 1;
    const sizes = merged.layerSizes ?? [d, Math.max(4, 2 * d), outputs];
    if (sizes.length < 2) throw new Error('MLPRegressor: layerSizes needs at least [inputs, outputs]');
    if (sizes[0] !== d) throw new Error(`MLPRegressor: layerSizes[0] is ${sizes[0]} but X has ${d} features`);
    if (sizes[sizes.length - 1] !== outputs) throw new Error(`MLPRegressor: the last layer size is ${sizes[sizes.length - 1]} but y has ${outputs} column${outputs === 1 ? '' : 's'}`);

    const x = nn.input(d);
    let h = x;
    for (const units of sizes.slice(1, -1)) {
      h = nn.dense(units, { activation: merged.activation })(h);
      if (merged.dropout > 0) h = nn.dropout(merged.dropout)(h);
    }
    const out = nn.dense(outputs)(h);
    this.model = nn.model(x, out, { loss: 'mse', normalizeY: merged.normalizeY, seed: merged.seed ?? undefined, name: 'MLPRegressor' });
    this.history = this.model.fitSync(dataX, dataY, {
      optimizer: merged.optimizer, learningRate: merged.learningRate, epochs: merged.epochs,
      batchSize: merged.batchSize, verbose: merged.verbose,
    });
    this.fitted = true;
    this.params = { ...merged, layerSizes: sizes, seed: this.model.seed };
    return this;
  }

  /**
   * Predict: a flat array for one output, rows otherwise. With
   * `{ samples: k }` and a dropout rate, Monte Carlo dropout's
   * `{ mean, std, epistemic, aleatoric }`.
   */
  predict(X, options) {
    if (!this.fitted || !this.model) throw new Error('MLPRegressor: estimator not fitted.');
    let matrix = X;
    if (isSpec(X)) {
      matrix = prepareX({ columns: X.X || X.columns, data: X.data, omit_missing: X.omit_missing ?? this.params.omit_missing }).X;
    }
    return this.model.predict(matrix, options);
  }

  /** d prediction / d x at one row, on the target scale. */
  predictGradient(x) {
    if (!this.fitted || !this.model) throw new Error('MLPRegressor: estimator not fitted.');
    return this.model.predictGradient(x);
  }

  /** Mean squared error on the target scale. */
  evaluate(X, y) {
    const pred = this.predict(X);
    let s = 0;
    for (let i = 0; i < y.length; i++) {
      if (Array.isArray(y[i])) for (let j = 0; j < y[i].length; j++) s += (pred[i][j] - y[i][j]) ** 2 / y[i].length;
      else s += (pred[i] - y[i]) ** 2;
    }
    return s / y.length;
  }

  summary() {
    if (!this.fitted || !this.model) throw new Error('MLPRegressor: estimator not fitted.');
    const { loss, epochs, stopped } = this.history;
    return {
      epochs, layerSizes: this.params.layerSizes, finalLoss: loss[loss.length - 1], initialLoss: loss[0], losses: loss, stopped,
      network: this.model.summary(),
    };
  }

  toJSON() {
    return { __class__: 'MLPRegressor', params: this.getParams(), fitted: !!this.fitted, model: this.model ? this.model.toJSON() : null, history: this.history };
  }

  static fromJSON(obj = {}) {
    const inst = new MLPRegressor(obj.params || {});
    if (obj.model) {
      inst.model = nn.fromJSON(obj.model);
      inst.history = obj.history ?? null;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default MLPRegressor;
