/**
 * Canonical Correlation Analysis (CCA) estimator
 *
 * Provides a scikit-learn style interface around the functional CCA utilities.
 */

import { Transformer } from '../../core/estimators/estimator.js';
import * as ccaFn from '../cca.js';

export class CCA extends Transformer {
  constructor(params = {}) {
    super(params);
    this.model = null;
  }

  /**
   * Fit the CCA model on paired data sets X and Y.
   *
   * Accepts a positional numeric call (`fit(X, Y[, opts])`) or a declarative
   * `{ data, X, Y }` object (`fit({ data, X, Y, columnsX, columnsY })`).
   * @param {Array<Array<number>>|Object} X - First data matrix (n samples × p features), or a declarative `{ data, X, Y }` object
   * @param {Array<Array<number>>} [Y] - Second data matrix (n samples × q features), for the positional call form
   * @param {Object} [opts] - Fitting options (used for the positional call form)
   * @returns {this} The fitted estimator (for chaining)
   */
  fit(X, Y = null, opts = {}) {
    let result;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.Y || X.columnsX || X.columnsY)
    ) {
      const callOpts = { ...this.params, ...X };
      result = ccaFn.fit(callOpts);
    } else {
      const options = { ...this.params, ...opts };
      result = ccaFn.fit(X, Y, options);
    }

    this.model = result;
    this.fitted = true;
    return this;
  }

  /**
   * Project new X data onto the fitted X canonical variates.
   * @param {Array<Array<number>>|Object} X - X data matrix (n samples × p features), or a declarative `{ data, columns }` object
   * @param {Object} [opts] - Transform options
   * @returns {Array<Object>} Canonical score objects, one per row (keyed `cca1`, `cca2`, ...)
   */
  transformX(X, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transformX().');
    }
    return ccaFn.transformX(this.model, X, opts);
  }

  /**
   * Project new Y data onto the fitted Y canonical variates.
   * @param {Array<Array<number>>|Object} Y - Y data matrix (n samples × q features), or a declarative `{ data, columns }` object
   * @param {Object} [opts] - Transform options
   * @returns {Array<Object>} Canonical score objects, one per row (keyed `cca1`, `cca2`, ...)
   */
  transformY(Y, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transformY().');
    }
    return ccaFn.transformY(this.model, Y, opts);
  }

  /**
   * Project new X and Y data onto their fitted canonical variates.
   * @param {Array<Array<number>>|Object} X - X data matrix (n samples × p features), or a declarative `{ data, columns }` object
   * @param {Array<Array<number>>|Object} Y - Y data matrix (n samples × q features), or a declarative `{ data, columns }` object
   * @param {Object} [opts] - Transform options
   * @returns {Object} Object with `xScores` and `yScores` arrays of canonical score objects
   */
  transform(X, Y, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transform().');
    }
    return ccaFn.transform(this.model, X, Y, opts);
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before summary().');
    }
    return {
      nSamples: this.model.nSamples,
      nComponents: this.model.nComponents,
      correlations: this.model.correlations.slice()
    };
  }

  toJSON() {
    return {
      __class__: 'CCA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new CCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default CCA;
