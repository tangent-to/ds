/**
 * Machine Learning Data Preprocessing Utilities
 */

import * as table from '../core/table.js';
import { mean, stddev } from '../core/math.js';

function prepareTableMatrix(
  X,
  {
    fallbackColumns = null,
    fallbackNaOmit = true,
    requireColumnsMessage = 'columns are required when using table data',
  } = {},
) {
  if (!X || typeof X !== 'object' || !('data' in X)) {
    return null;
  }

  const columns = X.columns || fallbackColumns;
  if (!columns || !columns.length) {
    throw new Error(requireColumnsMessage);
  }

  const hasNaFlag = ('naOmit' in X) || ('omit_missing' in X);
  const naOmit = hasNaFlag ? table.normalizeNaOmit(X) : fallbackNaOmit;

  const prepared = table.prepareX({
    data: X.data,
    columns,
    naOmit,
    omit_missing: X.omit_missing,
  });

  return { prepared, naOmit };
}

// ============= StandardScaler =============

/**
 * Standardize features by removing mean and scaling to unit variance
 */
export class StandardScaler {
  constructor() {
    this.means = null;
    this.stds = null;
    this.nFeatures = null;
    this._tableColumns = null;
    this._tableNaOmit = true;
  }

  /**
   * Compute mean and standard deviation
   * @param {Array<Array<number>>|Object} X - Feature matrix or {data, columns}
   * @returns {StandardScaler} this
   */
  fit(X) {
    // Support table descriptor: { data, columns }
    const tableInput = prepareTableMatrix(X, {
      requireColumnsMessage: 'StandardScaler.fit: columns are required when using table data',
    });
    if (tableInput) {
      const { prepared, naOmit } = tableInput;

      this.nFeatures = prepared.columns.length;
      this.means = [];
      this.stds = [];

      for (let j = 0; j < this.nFeatures; j++) {
        const col = prepared.X.map((row) => row[j]);
        this.means.push(mean(col));
        this.stds.push(stddev(col, false));
      }

      this._tableColumns = prepared.columns.slice();
      this._tableNaOmit = naOmit;

      return this;
    }

    // Standard array API
    const n = X.length;
    this.nFeatures = X[0].length;

    this.means = [];
    this.stds = [];

    for (let j = 0; j < this.nFeatures; j++) {
      const col = X.map((row) => row[j]);
      this.means.push(mean(col));
      // Use population std (ddof=0) to match sklearn's StandardScaler
      this.stds.push(stddev(col, false));
    }

    return this;
  }

  /**
   * Standardize features
   * @param {Array<Array<number>>|Object} X - Feature matrix or {data, columns}
   * @returns {Array<Array<number>>|Object} Scaled features or {data, columns, X}
   */
  transform(X) {
    if (this.means === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }

    // Support table descriptor: { data, columns, encoders? }
    const tableInput = prepareTableMatrix(X, {
      fallbackColumns: this._tableColumns,
      fallbackNaOmit: this._tableNaOmit,
      requireColumnsMessage: 'StandardScaler.transform: columns are required when using table data',
    });
    if (tableInput) {
      const { prepared } = tableInput;
      const featureColumns = prepared.columns;
      const rows = prepared.rows;
      const encoders = X.encoders;

      const scaledMatrix = prepared.X.map((row) =>
        row.map((val, j) => {
          const std = this.stds[j] > 0 ? this.stds[j] : 1;
          return (val - this.means[j]) / std;
        })
      );

      const scaledData = rows.map((row) => {
        const newRow = { ...row };
        featureColumns.forEach((col, j) => {
          const std = this.stds[j] > 0 ? this.stds[j] : 1;
          newRow[col] = (row[col] - this.means[j]) / std;
        });
        return newRow;
      });

      const result = {
        data: scaledData,
        columns: featureColumns.slice(),
        X: scaledMatrix,
      };

      // Pass through encoders if provided
      if (encoders) {
        result.metadata = { encoders };
      }

      return result;
    }

    // Standard array API
    return X.map((row) =>
      row.map((val, j) => {
        const std = this.stds[j] > 0 ? this.stds[j] : 1;
        return (val - this.means[j]) / std;
      })
    );
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }

  /**
   * Inverse transform (unscale)
   * @param {Array<Array<number>>} X - Scaled features
   * @returns {Array<Array<number>>} Original scale features
   */
  inverseTransform(X) {
    if (this.means === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }

    return X.map((row) =>
      row.map((val, j) => {
        const std = this.stds[j] > 0 ? this.stds[j] : 1;
        return val * std + this.means[j];
      })
    );
  }
}

// ============= MinMaxScaler =============

/**
 * Scale features to a given range [min, max]
 */
export class MinMaxScaler {
  constructor({ featureRange = [0, 1] } = {}) {
    this.featureRange = featureRange;
    this.dataMin = null;
    this.dataMax = null;
    this.nFeatures = null;
    this._tableColumns = null;
    this._tableNaOmit = true;
  }

  /**
   * Compute min and max for scaling
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {MinMaxScaler} this
   */
  fit(X) {
    const tableInput = prepareTableMatrix(X, {
      requireColumnsMessage: 'MinMaxScaler.fit: columns are required when using table data',
    });
    const dataMatrix = tableInput ? tableInput.prepared.X : X;
    if (!Array.isArray(dataMatrix) || !dataMatrix.length) {
      throw new Error('MinMaxScaler.fit: expected a non-empty matrix');
    }

    this.nFeatures = dataMatrix[0].length;
    this.dataMin = [];
    this.dataMax = [];

    for (let j = 0; j < this.nFeatures; j++) {
      const col = dataMatrix.map((row) => row[j]);
      this.dataMin.push(Math.min(...col));
      this.dataMax.push(Math.max(...col));
    }

    if (tableInput) {
      this._tableColumns = tableInput.prepared.columns.slice();
      this._tableNaOmit = tableInput.naOmit;
    }

    return this;
  }

  /**
   * Scale features to range
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  transform(X) {
    if (this.dataMin === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }

    const tableInput = prepareTableMatrix(X, {
      fallbackColumns: this._tableColumns,
      fallbackNaOmit: this._tableNaOmit,
      requireColumnsMessage: 'MinMaxScaler.transform: columns are required when using table data',
    });

    const [minRange, maxRange] = this.featureRange;

    if (tableInput) {
      const { prepared } = tableInput;
      const scaledMatrix = prepared.X.map((row) =>
        row.map((val, j) => {
          const dataRange = this.dataMax[j] - this.dataMin[j];
          if (dataRange === 0) return minRange;
          const scaled = (val - this.dataMin[j]) / dataRange;
          return scaled * (maxRange - minRange) + minRange;
        })
      );

      const scaledData = prepared.rows.map((row, idx) => {
        const newRow = { ...row };
        prepared.columns.forEach((col, j) => {
          newRow[col] = scaledMatrix[idx][j];
        });
        return newRow;
      });

      const result = {
        data: scaledData,
        columns: prepared.columns.slice(),
        X: scaledMatrix,
      };

      if (X.encoders) {
        result.metadata = { encoders: X.encoders };
      }

      return result;
    }

    return X.map((row) =>
      row.map((val, j) => {
        const dataRange = this.dataMax[j] - this.dataMin[j];
        if (dataRange === 0) return minRange;
        const scaled = (val - this.dataMin[j]) / dataRange;
        return scaled * (maxRange - minRange) + minRange;
      })
    );
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }

  /**
   * Inverse transform
   * @param {Array<Array<number>>} X - Scaled features
   * @returns {Array<Array<number>>} Original scale features
   */
  inverseTransform(X) {
    if (this.dataMin === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }

    const [minRange, maxRange] = this.featureRange;

    return X.map((row) =>
      row.map((val, j) => {
        const dataRange = this.dataMax[j] - this.dataMin[j];
        if (dataRange === 0) return this.dataMin[j];
        const unscaled = (val - minRange) / (maxRange - minRange);
        return unscaled * dataRange + this.dataMin[j];
      })
    );
  }
}

// ============= LabelEncoder =============

/**
 * Encode target labels with value between 0 and n_classes-1
 */
export class LabelEncoder {
  constructor() {
    this.classes = null;
    this.classMap = null;
    this._tableColumn = null;
  }

  _extractLabelVector(input, { fallbackColumn = null, forTransform = false } = {}) {
    if (Array.isArray(input)) {
      return { values: input, rows: null, column: null };
    }

    if (input && typeof input === 'object' && 'data' in input) {
      const column = input.column || fallbackColumn;
      if (!column) {
        throw new Error(
          forTransform
            ? 'LabelEncoder.transform: column is required when using table data'
            : 'LabelEncoder.fit: column is required when using table data',
        );
      }
      const rows = table.normalize(input.data);
      const values = rows.map((row) => row[column]);
      return { values, rows, column };
    }

    throw new Error('LabelEncoder expects an array or { data, column } descriptor');
  }

  /**
   * Fit label encoder
   * @param {Array} y - Target labels
   * @returns {LabelEncoder} this
   */
  fit(y) {
    const { values, column } = this._extractLabelVector(y, { fallbackColumn: this._tableColumn });
    this.classes = [...new Set(values)].sort();
    this.classMap = new Map(this.classes.map((c, i) => [c, i]));
    this._tableColumn = column || null;
    return this;
  }

  /**
   * Transform labels to indices
   * @param {Array} y - Target labels
   * @returns {Array<number>} Encoded labels
   */
  transform(y) {
    if (this.classMap === null) {
      throw new Error('LabelEncoder not fitted. Call fit() first.');
    }

    const { values, rows, column } = this._extractLabelVector(
      y,
      { fallbackColumn: this._tableColumn, forTransform: true },
    );

    const encoded = values.map((label) => {
      if (!this.classMap.has(label)) {
        throw new Error(`Unknown label: ${label}`);
      }
      return this.classMap.get(label);
    });

    if (!rows) {
      return encoded;
    }

    const encodedRows = rows.map((row, idx) => ({
      ...row,
      [column]: encoded[idx],
    }));

    return {
      data: encodedRows,
      values: encoded,
      column,
    };
  }

  /**
   * Fit and transform in one step
   * @param {Array} y - Target labels
   * @returns {Array<number>} Encoded labels
   */
  fitTransform(y) {
    return this.fit(y).transform(y);
  }

  /**
   * Transform indices back to original labels
   * @param {Array<number>} y - Encoded labels
   * @returns {Array} Original labels
   */
  inverseTransform(y) {
    if (this.classes === null) {
      throw new Error('LabelEncoder not fitted. Call fit() first.');
    }

    return y.map((idx) => {
      if (idx < 0 || idx >= this.classes.length) {
        throw new Error(`Invalid index: ${idx}`);
      }
      return this.classes[idx];
    });
  }
}

// ============= OneHotEncoder =============

/**
 * Encode categorical features as one-hot numeric array
 */
export class OneHotEncoder {
  constructor() {
    this.categories = null;
    this.nFeatures = null;
    this._tableColumns = null;
  }

  _prepareInput(X, {
    fallbackColumns = null,
    requireColumnsMessage = 'OneHotEncoder: columns are required when using table data',
  } = {}) {
    if (Array.isArray(X)) {
      if (!X.length || !Array.isArray(X[0])) {
        throw new Error('OneHotEncoder expects a 2D array');
      }
      return { matrix: X, rows: null, columns: null };
    }

    if (X && typeof X === 'object' && 'data' in X) {
      const columns = X.columns || fallbackColumns;
      if (!columns || (Array.isArray(columns) && columns.length === 0)) {
        throw new Error(requireColumnsMessage);
      }
      const columnList = Array.isArray(columns) ? columns : [columns];
      const rows = table.normalize(X.data);
      const matrix = rows.map((row) => columnList.map((col) => row[col]));
      return {
        matrix,
        rows,
        columns: columnList,
      };
    }

    throw new Error('OneHotEncoder expects a 2D array or { data, columns } descriptor');
  }

  /**
   * Fit encoder by discovering categories
   * @param {Array<Array>|Object} X - Categorical features matrix or {data, columns}
   * @returns {OneHotEncoder} this
   */
  fit(X) {
    const { matrix, columns } = this._prepareInput(X, {
      requireColumnsMessage: 'OneHotEncoder.fit: columns are required when using table data',
    });

    this.nFeatures = matrix[0].length;
    this.categories = [];

    for (let j = 0; j < this.nFeatures; j++) {
      const col = matrix.map((row) => row[j]);
      const unique = [...new Set(col)].sort();
      this.categories.push(unique);
    }

    this._tableColumns = columns ? columns.slice() : null;

    return this;
  }

  /**
   * Transform to one-hot encoding
   * @param {Array<Array>} X - Categorical features
   * @returns {Array<Array<number>>} One-hot encoded features
   */
  transform(X) {
    if (this.categories === null) {
      throw new Error('OneHotEncoder not fitted. Call fit() first.');
    }

    const { matrix, rows, columns } = this._prepareInput(X, {
      fallbackColumns: this._tableColumns,
      requireColumnsMessage: 'OneHotEncoder.transform: columns are required when using table data',
    });

    const encoded = matrix.map((row) => {
      const encodedRow = [];

      for (let j = 0; j < this.nFeatures; j++) {
        const value = row[j];
        const categories = this.categories[j];
        const idx = categories.indexOf(value);

        if (idx === -1) {
          throw new Error(`Unknown category: ${value} in feature ${j}`);
        }

        for (let k = 0; k < categories.length; k++) {
          encodedRow.push(k === idx ? 1 : 0);
        }
      }

      return encodedRow;
    });

    if (!rows) {
      return encoded;
    }

    const baseNames = columns || this._tableColumns || Array.from(
      { length: this.nFeatures },
      (_, idx) => `feature${idx}`,
    );

    const perFeatureNames = this.categories.map((cats, featureIdx) => {
      const prefix = baseNames && baseNames[featureIdx] ? `${baseNames[featureIdx]}_` : '';
      return cats.map((cat) => `${prefix}${String(cat)}`);
    });

    const flatNames = perFeatureNames.flat();

    const transformedRows = rows.map((row, rowIdx) => {
      const newRow = { ...row };
      let offset = 0;
      perFeatureNames.forEach((names, featureIdx) => {
        names.forEach((name, idx) => {
          newRow[name] = encoded[rowIdx][offset + idx];
        });
        if (baseNames && baseNames[featureIdx]) {
          delete newRow[baseNames[featureIdx]];
        }
        offset += names.length;
      });
      return newRow;
    });

    return {
      data: transformedRows,
      columns: flatNames,
      X: encoded,
    };
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array>} X - Categorical features
   * @returns {Array<Array<number>>} One-hot encoded features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }

  /**
   * Get feature names after one-hot encoding
   * @returns {Array<string>} Feature names
   */
  getFeatureNames() {
    if (this.categories === null) {
      throw new Error('OneHotEncoder not fitted. Call fit() first.');
    }

    const names = [];
    for (let j = 0; j < this.nFeatures; j++) {
      for (const category of this.categories[j]) {
        names.push(`feature${j}_${category}`);
      }
    }
    return names;
  }
}

// ============= PolynomialFeatures =============

/**
 * Generate polynomial and interaction features
 */
export class PolynomialFeatures {
  constructor({ degree = 2, includeBias = false } = {}) {
    this.degree = degree;
    this.includeBias = includeBias;
    this.nInputFeatures = null;
    this.nOutputFeatures = null;
    this._featurePatterns = [];
    this._tableColumns = null;
    this._tableNaOmit = true;
  }

  _normalizeInput(X, {
    fallbackColumns = null,
    fallbackNaOmit = true,
    requireColumnsMessage = 'PolynomialFeatures: columns are required when using table data',
  } = {}) {
    const tableInput = prepareTableMatrix(X, {
      fallbackColumns,
      fallbackNaOmit,
      requireColumnsMessage,
    });

    if (tableInput) {
      return {
        matrix: tableInput.prepared.X,
        tableInput,
      };
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error(requireColumnsMessage);
    }

    return { matrix: X, tableInput: null };
  }

  _buildFeaturePatterns() {
    this._featurePatterns = [];

    if (this.includeBias) {
      this._featurePatterns.push({ type: 'bias' });
    }

    // Degree 1 terms (original features)
    for (let i = 0; i < this.nInputFeatures; i++) {
      this._featurePatterns.push({ type: 'monomial', indices: [i] });
    }

    for (let d = 2; d <= this.degree; d++) {
      this._appendDegreePatterns(d);
    }
  }

  _appendDegreePatterns(degree) {
    const combo = [];
    const n = this.nInputFeatures;

    const generate = (start, depth) => {
      if (depth === degree) {
        this._featurePatterns.push({ type: 'monomial', indices: combo.slice() });
        return;
      }
      for (let i = start; i < n; i++) {
        combo.push(i);
        generate(i, depth + 1);
        combo.pop();
      }
    };

    generate(0, 0);
  }

  _evaluatePattern(pattern, row) {
    if (pattern.type === 'bias') {
      return 1;
    }

    return pattern.indices.reduce((product, idx) => product * row[idx], 1);
  }

  _buildFeatureNames(columns = null) {
    const baseNames = columns && columns.length === this.nInputFeatures
      ? columns
      : Array.from({ length: this.nInputFeatures }, (_, i) => `x${i + 1}`);

    return this._featurePatterns.map((pattern) => {
      if (pattern.type === 'bias') {
        return 'bias';
      }
      const counts = new Map();
      pattern.indices.forEach((idx) => {
        counts.set(idx, (counts.get(idx) || 0) + 1);
      });
      const parts = [];
      for (const [idx, count] of counts.entries()) {
        const name = baseNames[idx];
        parts.push(count > 1 ? `${name}^${count}` : name);
      }
      return parts.join('*');
    });
  }

  /**
   * Fit by determining input/output dimensions
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {PolynomialFeatures} this
   */
  fit(X) {
    const { matrix, tableInput } = this._normalizeInput(X, {
      requireColumnsMessage: 'PolynomialFeatures.fit: columns are required when using table data',
    });

    if (!matrix.length) {
      throw new Error('PolynomialFeatures.fit: expected non-empty input');
    }

    this.nInputFeatures = matrix[0].length;
    this._buildFeaturePatterns();
    this.nOutputFeatures = this._featurePatterns.length;

    if (tableInput) {
      this._tableColumns = tableInput.prepared.columns.slice();
      this._tableNaOmit = tableInput.naOmit;
    }

    return this;
  }

  /**
   * Transform to polynomial features
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Polynomial features
   */
  transform(X) {
    if (this.nInputFeatures === null) {
      throw new Error('PolynomialFeatures not fitted. Call fit() first.');
    }

    const { matrix, tableInput } = this._normalizeInput(X, {
      fallbackColumns: this._tableColumns,
      fallbackNaOmit: this._tableNaOmit,
      requireColumnsMessage:
        'PolynomialFeatures.transform: columns are required when using table data',
    });

    if (!matrix.length) {
      throw new Error('PolynomialFeatures.transform: expected non-empty input');
    }
    if (matrix[0].length !== this.nInputFeatures) {
      throw new Error('PolynomialFeatures.transform: input feature size mismatch');
    }

    const transformed = matrix.map((row) =>
      this._featurePatterns.map((pattern) => this._evaluatePattern(pattern, row))
    );

    if (!tableInput) {
      return transformed;
    }

    const columnNames = this._buildFeatureNames(tableInput.prepared.columns);
    const transformedRows = tableInput.prepared.rows.map((row, idx) => {
      const newRow = { ...row };
      columnNames.forEach((name, colIdx) => {
        newRow[name] = transformed[idx][colIdx];
      });
      return newRow;
    });

    return {
      data: transformedRows,
      columns: columnNames,
      X: transformed,
    };
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Polynomial features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
}

// ============= Higher-level preprocessing functions =============

/**
 * Convert string columns to numeric
 * Useful when CSV parsers incorrectly infer column types
 *
 * @param {Object} options
 * @param {Array|Object} options.data - Input data
 * @param {Array<string>} options.columns - Column names to convert
 * @returns {Array<Object>} Data with converted columns
 */
export function parseNumeric({ data, columns = [] }) {
  const rows = table.normalize(data);

  return rows.map((row) => {
    const newRow = { ...row };
    for (const col of columns) {
      if (col in newRow && newRow[col] !== null && newRow[col] !== undefined) {
        const val = newRow[col];
        if (typeof val === 'string') {
          const parsed = parseFloat(val);
          newRow[col] = isNaN(parsed) ? val : parsed;
        }
      }
    }
    return newRow;
  });
}

/**
 * Clean and validate categorical columns
 * Removes rows with invalid categories
 *
 * @param {Object} options
 * @param {Array|Object} options.data - Input data
 * @param {Object} options.validCategories - Map of column names to arrays of valid values
 * @returns {Object} { data: cleaned data, removed: count of removed rows }
 */
export function cleanCategorical({ data, validCategories = {} }) {
  const rows = table.normalize(data);

  const cleaned = rows.filter((row) => {
    for (const [column, validValues] of Object.entries(validCategories)) {
      const value = row[column];
      if (value === null || value === undefined) return false;
      if (!validValues.includes(value)) return false;
    }
    return true;
  });

  return {
    data: cleaned,
    removed: rows.length - cleaned.length,
  };
}

/**
 * Label encode a categorical column
 * Maps categories to integers (0, 1, 2, ...)
 *
 * @param {Object} options
 * @param {Array|Object} options.data - Input data
 * @param {string} options.column - Column to encode
 * @param {string} [options.outputColumn] - Name for encoded column (default: column + '_idx')
 * @param {boolean} [options.keepOriginal=true] - Keep original column
 * @param {Array} [options.categories] - Predefined category order (optional)
 * @returns {Object} { data, encoder, outputColumn }
 */
export function labelEncode({
  data,
  column,
  outputColumn,
  keepOriginal = true,
  categories = null,
}) {
  const rows = table.normalize(data);
  const encoder = new table.LabelEncoder();

  const values = rows.map((row) => row[column]);

  if (categories) {
    encoder.classes_ = categories.slice();
    encoder.classIndex = new Map(categories.map((c, i) => [c, i]));
  } else {
    encoder.fit(values);
  }

  const encoded = encoder.transform(values);
  const outCol = outputColumn || `${column}_idx`;

  const result = rows.map((row, i) => {
    const newRow = { ...row };
    newRow[outCol] = encoded[i];
    if (!keepOriginal) {
      delete newRow[column];
    }
    return newRow;
  });

  return {
    data: result,
    encoder,
    outputColumn: outCol,
  };
}

/**
 * Declarative preprocessing pipeline for tabular data
 * Handles numeric parsing, data validation, and categorical encoding in one step
 *
 * @param {Object} options
 * @param {Array|Object} options.data - Input data (array of objects or Arquero table)
 * @param {Array<string>} [options.parseNumeric] - Column names to convert from string to numeric
 * @param {Object} [options.validCategories] - Validation rules for categorical columns (removes invalid rows)
 * @param {Array<Object>} [options.labelEncode] - Columns to label encode: [{ column, outputColumn?, categories? }]
 * @param {Array<Object>} [options.oneHotEncode] - Columns to one-hot encode: [{ columns, dropFirst?, prefix? }]
 * @param {boolean} [options.verbose=true] - Print preprocessing info
 * @returns {Object} { data, info: { parsed, cleaned, labelEncoders, oneHotInfo } }
 */
export function preprocess({
  data,
  parseNumeric: parseNumericCols = null,
  validCategories = null,
  labelEncode: labelEncodeSpecs = [],
  oneHotEncode: oneHotEncodeSpecs = [],
  verbose = true,
}) {
  let current = data;
  const info = {
    parsed: null,
    cleaned: null,
    labelEncoders: {},
    oneHotInfo: new Map(),
  };

  // Step 0: Parse numeric columns if needed
  if (parseNumericCols && parseNumericCols.length > 0) {
    current = parseNumeric({ data: current, columns: parseNumericCols });
    info.parsed = { columns: parseNumericCols };

    if (verbose) {
      console.log(
        `✓ Parsed ${parseNumericCols.length} numeric columns: ${parseNumericCols.join(', ')}`,
      );
    }
  }

  // Step 1: Clean data
  if (validCategories) {
    const cleaned = cleanCategorical({ data: current, validCategories });
    current = cleaned.data;
    info.cleaned = {
      originalRows: table.normalize(data).length,
      cleanedRows: table.normalize(current).length,
      removed: cleaned.removed,
    };

    if (verbose && cleaned.removed > 0) {
      console.log(
        `✓ Removed ${cleaned.removed} invalid rows (${info.cleaned.cleanedRows} remaining)`,
      );
    }
  }

  // Step 2: Label encoding
  for (const spec of labelEncodeSpecs) {
    const result = labelEncode({
      data: current,
      column: spec.column,
      outputColumn: spec.outputColumn,
      keepOriginal: spec.keepOriginal !== false,
      categories: spec.categories,
    });
    current = result.data;
    info.labelEncoders[spec.column] = {
      encoder: result.encoder,
      outputColumn: result.outputColumn,
      categories: result.encoder.classes_,
    };

    if (verbose) {
      console.log(
        `✓ Label encoded '${spec.column}' → '${result.outputColumn}' (${result.encoder.classes_.length} categories)`,
      );
    }
  }

  // Step 3: One-hot encoding
  for (const spec of oneHotEncodeSpecs) {
    const result = table.oneHotEncodeTable({
      data: current,
      columns: spec.columns,
      dropFirst: spec.dropFirst !== undefined ? spec.dropFirst : true,
      keepOriginal: spec.keepOriginal || false,
      prefix: spec.prefix !== undefined ? spec.prefix : true,
    });
    current = result.data;

    // Store info for each encoded column
    for (const [col, colInfo] of result.dummyInfo.entries()) {
      info.oneHotInfo.set(col, colInfo);

      if (verbose) {
        const nCols = colInfo.columnNames.length;
        const nCats = colInfo.categories.length;
        console.log(
          `✓ One-hot encoded '${col}' → ${nCols} columns (${nCats} categories, dropFirst=${colInfo.dropFirst})`,
        );
      }
    }
  }

  return {
    data: current,
    info,
  };
}

/**
 * Fit a preprocessing pipeline and store the transformers
 * Use this on training data, then apply the same transformers to test data
 *
 * @param {Object} options - Same as preprocessCategorical
 * @returns {Object} { data, pipeline: reusable pipeline object }
 */
export function fitPreprocessor(options) {
  const result = preprocess(options);

  const pipeline = {
    parseNumeric: options.parseNumeric || null,
    validCategories: options.validCategories,
    labelEncoders: result.info.labelEncoders,
    oneHotEncoders: new Map(),
    labelEncodeSpecs: options.labelEncode || [],
    oneHotEncodeSpecs: options.oneHotEncode || [],
  };

  // Extract encoders from oneHotInfo
  for (const [col, info] of result.info.oneHotInfo.entries()) {
    pipeline.oneHotEncoders.set(col, info.encoder);
  }

  return {
    data: result.data,
    pipeline,
    info: result.info,
  };
}

/**
 * Transform new data using a fitted preprocessing pipeline
 *
 * @param {Object} options
 * @param {Array|Object} options.data - New data to transform
 * @param {Object} options.pipeline - Pipeline from fitPreprocessor
 * @param {boolean} [options.verbose=false] - Print info
 * @returns {Object} { data }
 */
export function transformWithPipeline({ data, pipeline, verbose = false }) {
  let current = data;

  // Step 0: Parse numeric columns if needed
  if (pipeline.parseNumeric && pipeline.parseNumeric.length > 0) {
    current = parseNumeric({ data: current, columns: pipeline.parseNumeric });

    if (verbose) {
      console.log(
        `✓ Parsed ${pipeline.parseNumeric.length} numeric columns: ${
          pipeline.parseNumeric.join(', ')
        }`,
      );
    }
  }

  // Step 1: Clean data (using same validation rules)
  if (pipeline.validCategories) {
    const cleaned = cleanCategorical({
      data: current,
      validCategories: pipeline.validCategories,
    });
    current = cleaned.data;

    if (verbose && cleaned.removed > 0) {
      console.log(`✓ Removed ${cleaned.removed} invalid rows`);
    }
  }

  // Step 2: Apply label encoders
  for (const spec of pipeline.labelEncodeSpecs) {
    const encoderInfo = pipeline.labelEncoders[spec.column];
    const rows = table.normalize(current);
    const values = rows.map((row) => row[spec.column]);
    const encoded = encoderInfo.encoder.transform(values);

    current = rows.map((row, i) => {
      const newRow = { ...row };
      newRow[encoderInfo.outputColumn] = encoded[i];
      if (spec.keepOriginal === false) {
        delete newRow[spec.column];
      }
      return newRow;
    });

    if (verbose) {
      console.log(`✓ Label encoded '${spec.column}' → '${encoderInfo.outputColumn}'`);
    }
  }

  // Step 3: Apply one-hot encoders
  for (const spec of pipeline.oneHotEncodeSpecs) {
    const columns = Array.isArray(spec.columns) ? spec.columns : [spec.columns];

    for (const col of columns) {
      const encoder = pipeline.oneHotEncoders.get(col);
      if (!encoder) continue;

      const rows = table.normalize(current);
      const values = rows.map((row) => row[col]);
      const encodedVecs = encoder.transform(values);

      const dropFirst = spec.dropFirst !== undefined ? spec.dropFirst : true;
      const prefix = spec.prefix !== undefined ? spec.prefix : true;

      const categories = encoder.categories_;
      const includedIndices = [];
      const columnNames = [];

      categories.forEach((cat, idx) => {
        if (dropFirst && idx === 0) return;
        includedIndices.push(idx);
        const colName = prefix ? `${col}_${String(cat)}` : String(cat);
        columnNames.push(colName);
      });

      current = rows.map((row, rowIdx) => {
        const newRow = { ...row };
        const encodedVec = encodedVecs[rowIdx];
        includedIndices.forEach((catIdx, idx) => {
          const colName = columnNames[idx];
          newRow[colName] = encodedVec[catIdx];
        });
        if (!spec.keepOriginal) {
          delete newRow[col];
        }
        return newRow;
      });

      if (verbose) {
        console.log(`✓ One-hot encoded '${col}' → ${columnNames.length} columns`);
      }
    }
  }

  return { data: current };
}

// Backward compatibility alias
export const preprocessCategorical = preprocess;

export { trainTestSplit } from './validation.js';
