/**
 * Recipe pattern for inspectable data preprocessing pipelines
 *
 * The Recipe API provides a chainable, declarative way to define preprocessing workflows.
 * Unlike traditional pipelines, Recipe allows full inspection of intermediate results
 * while ensuring transformers are correctly fitted on training data and applied to test/new data.
 *
 * Key features:
 * - Chainable methods for defining preprocessing steps
 * - prep() executes all steps and returns inspectable results
 * - bake() applies fitted transformers to new data
 * - All intermediate outputs available for inspection
 * - Prevents data leakage (test data never used for fitting)
 *
 * @example
 * // Define a preprocessing recipe
 * const recipe = ds.ml.recipe({
 *   data: myData,
 *   X: ['feature1', 'feature2', 'category'],
 *   y: 'target'
 * })
 *   .oneHot(['category'])
 *   .scale(['feature1', 'feature2'])
 *   .split({ ratio: 0.7, shuffle: true });
 *
 * // Execute recipe
 * const prepped = recipe.prep();
 *
 * // Inspect intermediate steps
 * console.log(prepped.steps[0].output);  // After one-hot encoding
 * console.log(prepped.transformers.scale);  // Fitted scaler
 *
 * // Train model
 * const model = new ds.ml.KNNClassifier({ k: 3 })
 *   .fit({ data: prepped.train.data, X: prepped.train.X, y: prepped.train.y });
 *
 * // Apply to new data
 * const newPrepped = recipe.bake(newData);
 * const predictions = model.predict({ data: newPrepped.data, X: newPrepped.X });
 */

import * as table from '../core/table.js';
import { trainTestSplit as splitData } from './validation.js';
import { MinMaxScaler, StandardScaler } from './preprocessing.js';
import * as pca from '../mva/pca.js';
import * as lda from '../mva/lda.js';
import * as rda from '../mva/rda.js';

/**
 * Create a preprocessing recipe
 *
 * Factory function to create a new Recipe instance. The recipe defines a sequence
 * of preprocessing operations that will be applied to data.
 *
 * @param {Object} options - Initial data descriptor
 * @param {Array<Object>} options.data - Input data as array of objects
 * @param {Array<string>|string} options.X - Feature column names
 * @param {string} options.y - Target column name
 * @returns {Recipe} A recipe object for chaining preprocessing steps
 *
 * @example
 * const recipe = ds.ml.recipe({
 *   data: penguins,
 *   X: ['bill_length', 'bill_depth', 'flipper_length'],
 *   y: 'species'
 * });
 */
export function recipe({ data, X, y }) {
  return new Recipe({ data, X, y });
}

/**
 * Recipe class for building inspectable preprocessing workflows
 *
 * A Recipe defines a sequence of data preprocessing steps that can be:
 * 1. Defined declaratively through chainable methods
 * 2. Executed with prep() to fit transformers on training data
 * 3. Applied to new data with bake() using fitted transformers
 * 4. Inspected at every step to understand transformations
 *
 * Supported preprocessing operations:
 * - parseNumeric(): Convert string columns to numbers
 * - clean(): Remove rows with invalid categorical values
 * - oneHot(): One-hot encode categorical columns
 * - scale(): Scale numeric columns (standard or minmax)
 * - pca(): Principal Component Analysis for dimensionality reduction
 * - lda(): Linear Discriminant Analysis for supervised dimensionality reduction
 * - rda(): Redundancy Analysis for constrained ordination
 * - split(): Split into train/test sets
 *
 * @example
 * // Complete workflow
 * const recipe = new Recipe({ data: myData, X: features, y: 'target' })
 *   .parseNumeric(['age', 'price'])
 *   .clean({ category: ['A', 'B', 'C'] })
 *   .oneHot(['category'])
 *   .scale(['age', 'price'], { method: 'standard' })
 *   .split({ ratio: 0.7, seed: 42 });
 *
 * const result = recipe.prep();
 * // result.train.data - training data
 * // result.test.data - test data
 * // result.transformers - fitted transformers (scale, oneHot, etc.)
 * // result.steps - intermediate outputs for inspection
 */
export class Recipe {
  constructor({ data, X, y }) {
    this.initialData = data;
    this.X = Array.isArray(X) ? X : [X];
    this.y = y;
    this.steps = [];
    this._prepared = false;
    this._transformers = {};
  }

  /**
   * Parse string columns as numeric
   *
   * Converts string representations of numbers to actual numeric values.
   * Useful when CSV parsers incorrectly infer column types.
   *
   * @param {Array<string>} columns - Column names to parse
   * @returns {Recipe} this (for chaining)
   *
   * @example
   * recipe.parseNumeric(['age', 'price', 'quantity']);
   */
  parseNumeric(columns) {
    this.steps.push({
      name: 'parseNumeric',
      type: 'transform',
      columns,
      fn: (data) => {
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
      },
    });
    return this;
  }

  /**
   * Clean categorical columns
   * @param {Object} validCategories - Map of column -> valid values
   * @returns {Recipe} this
   */
  clean(validCategories) {
    this.steps.push({
      name: 'clean',
      type: 'filter',
      validCategories,
      fn: (data) => {
        const rows = table.normalize(data);
        return rows.filter((row) => {
          for (const [column, validValues] of Object.entries(validCategories)) {
            const value = row[column];
            if (value === null || value === undefined) return false;
            if (!validValues.includes(value)) return false;
          }
          return true;
        });
      },
    });
    return this;
  }

  /**
   * One-hot encode categorical columns
   * @param {Array<string>} columns - Columns to encode
   * @param {Object} options - Encoding options
   * @param {boolean} options.dropFirst - Drop first category
   * @param {boolean} options.prefix - Use column name prefix
   * @returns {Recipe} this
   */
  oneHot(columns, { dropFirst = true, prefix = true } = {}) {
    this.steps.push({
      name: 'oneHot',
      type: 'encode',
      columns,
      dropFirst,
      prefix,
      transformer: null, // Will be fitted during prep()
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit: discover categories and create encoder
          const result = table.oneHotEncodeTable({
            data,
            columns,
            dropFirst,
            keepOriginal: false,
            prefix,
          });
          return {
            data: result.data,
            transformer: result.dummyInfo,
          };
        } else {
          // Transform: use fitted encoder
          const rows = table.normalize(data);
          const transformedRows = rows.map((row) => {
            const newRow = { ...row };

            // For each column that was one-hot encoded
            for (const col of columns) {
              const colInfo = transformer.get(col);
              if (!colInfo) continue;

              const value = row[col];

              // Create dummy columns based on fitted categories
              for (const dummyCol of colInfo.columnNames) {
                // Extract the category from the column name
                const category = prefix
                  ? dummyCol.substring(col.length + 1) // Remove "col_" prefix
                  : dummyCol;

                // Set 1 if this is the value, 0 otherwise
                newRow[dummyCol] = (String(value) === category) ? 1 : 0;
              }

              // Remove original column
              delete newRow[col];
            }

            return newRow;
          });

          return {
            data: transformedRows,
            transformer,
          };
        }
      },
    });
    return this;
  }

  /**
   * Scale numeric columns
   * @param {Array<string>} columns - Columns to scale
   * @param {Object} options - Scaling options
   * @param {string} options.method - 'standard' or 'minmax'
   * @returns {Recipe} this
   */
  scale(columns, { method = 'standard' } = {}) {
    this.steps.push({
      name: 'scale',
      type: 'scale',
      columns,
      method,
      transformer: null, // Will be fitted during prep()
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit on training data
          const scaler = method === 'minmax' ? new MinMaxScaler() : new StandardScaler();

          scaler.fit({ data, columns });
          return {
            data: scaler.transform({ data, columns }).data,
            transformer: scaler,
          };
        } else {
          // Transform using fitted scaler
          return {
            data: transformer.transform({ data, columns }).data,
            transformer,
          };
        }
      },
    });
    return this;
  }

  /**
   * Apply Principal Component Analysis for dimensionality reduction
   *
   * Reduces the dimensionality of numeric features by projecting them onto
   * principal components. This is a feature extraction/transformation step.
   *
   * @param {Object} options - PCA options
   * @param {Array<string>} options.columns - Columns to include in PCA
   * @param {number} [options.nComponents] - Number of components to keep (default: all)
   * @param {boolean} [options.scale=true] - Scale features before PCA
   * @param {boolean} [options.center=true] - Center features before PCA
   * @returns {Recipe} this (for chaining)
   *
   * @example
   * recipe.pca({
   *   columns: ['feature1', 'feature2', 'feature3'],
   *   nComponents: 2,
   *   scale: true
   * });
   */
  pca({ columns, nComponents = null, scale = true, center = true } = {}) {
    this.steps.push({
      name: 'pca',
      type: 'dimreduction',
      columns,
      nComponents,
      scale,
      center,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit PCA on training data
          const model = pca.fit({ data, columns, scale, center });

          // Extract numeric matrix for transform (PCA transform expects array format)
          const rows = table.normalize(data);
          const X = rows.map((row) => columns.map((col) => row[col]));

          // Transform data
          const scoreObjects = pca.transform(model, X);

          // Create new data with PC scores
          const pcColumns = Array.from(
            { length: nComponents || model.eigenvalues.length },
            (_, i) => `PC${i + 1}`,
          );

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove original columns
            columns.forEach((col) => delete newRow[col]);
            // Add PC scores (scoreObjects are objects with pc1, pc2, etc.)
            pcColumns.forEach((pc, j) => {
              const pcKey = `pc${j + 1}`;
              newRow[pc] = scoreObjects[i][pcKey];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer: { model, pcColumns },
          };
        } else {
          // Transform using fitted PCA
          const rows = table.normalize(data);
          const X = rows.map((row) => columns.map((col) => row[col]));
          const scoreObjects = pca.transform(transformer.model, X);

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove original columns
            columns.forEach((col) => delete newRow[col]);
            // Add PC scores (scoreObjects are objects with pc1, pc2, etc.)
            transformer.pcColumns.forEach((pc, j) => {
              const pcKey = `pc${j + 1}`;
              newRow[pc] = scoreObjects[i][pcKey];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer,
          };
        }
      },
    });

    // Update feature list
    const pcColumns = Array.from(
      { length: nComponents || columns.length },
      (_, i) => `PC${i + 1}`,
    );
    this.X = this.X.filter((col) => !columns.includes(col)).concat(pcColumns);

    return this;
  }

  /**
   * Apply Linear Discriminant Analysis for supervised dimensionality reduction
   *
   * Reduces dimensionality while maximizing class separation. Requires a target
   * variable. This is both feature extraction and supervised learning.
   *
   * @param {Object} options - LDA options
   * @param {Array<string>} options.columns - Columns to include in LDA
   * @param {number} [options.nComponents] - Number of discriminants to keep
   * @param {boolean} [options.scale=false] - Scale features before LDA
   * @returns {Recipe} this (for chaining)
   *
   * @example
   * recipe.lda({
   *   columns: ['feature1', 'feature2', 'feature3'],
   *   nComponents: 2
   * });
   */
  lda({ columns, nComponents = null, scale = false } = {}) {
    if (!this.y) {
      throw new Error('LDA requires a target variable (y). Specify y in recipe().');
    }

    this.steps.push({
      name: 'lda',
      type: 'dimreduction',
      columns,
      nComponents,
      scale,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit LDA on training data (supports table API)
          const model = lda.fit({ data, X: columns, y: this.y, scale, scaling: nComponents });

          // Extract numeric matrix for transform (LDA transform expects array format)
          const rows = table.normalize(data);
          const X = rows.map((row) => columns.map((col) => row[col]));
          const scoreObjects = lda.transform(model, X);

          // Create new data with LD scores
          const ldColumns = Array.from(
            { length: nComponents || model.scaling.length },
            (_, i) => `LD${i + 1}`,
          );

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove original columns
            columns.forEach((col) => delete newRow[col]);
            // Add LD scores (scoreObjects are objects with ld1, ld2, etc.)
            ldColumns.forEach((ld, j) => {
              const ldKey = `ld${j + 1}`;
              newRow[ld] = scoreObjects[i][ldKey];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer: { model, ldColumns },
          };
        } else {
          // Transform using fitted LDA
          const rows = table.normalize(data);
          const X = rows.map((row) => columns.map((col) => row[col]));
          const scoreObjects = lda.transform(transformer.model, X);

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove original columns
            columns.forEach((col) => delete newRow[col]);
            // Add LD scores (scoreObjects are objects with ld1, ld2, etc.)
            transformer.ldColumns.forEach((ld, j) => {
              const ldKey = `ld${j + 1}`;
              newRow[ld] = scoreObjects[i][ldKey];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer,
          };
        }
      },
    });

    // Update feature list
    const ldColumns = Array.from(
      { length: nComponents || Math.min(columns.length, 10) }, // Default max
      (_, i) => `LD${i + 1}`,
    );
    this.X = this.X.filter((col) => !columns.includes(col)).concat(ldColumns);

    return this;
  }

  /**
   * Apply Redundancy Analysis for constrained ordination
   *
   * RDA combines regression and PCA to find patterns in response variables
   * that are explained by predictor variables. Useful for ecological data.
   *
   * @param {Object} options - RDA options
   * @param {Array<string>} options.response - Response variable columns
   * @param {Array<string>} options.predictors - Predictor variable columns
   * @param {number} [options.nComponents] - Number of RDA axes to keep
   * @param {boolean} [options.scale=false] - Scale variables before RDA
   * @returns {Recipe} this (for chaining)
   *
   * @example
   * recipe.rda({
   *   response: ['species1', 'species2', 'species3'],
   *   predictors: ['temperature', 'rainfall'],
   *   nComponents: 2
   * });
   */
  rda({ response, predictors, nComponents = null, scale = false } = {}) {
    this.steps.push({
      name: 'rda',
      type: 'dimreduction',
      response,
      predictors,
      nComponents,
      scale,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit RDA on training data
          const model = rda.fit({ data, response, predictors, scale });

          // Transform data
          const scores = rda.transform(model, { data, response, predictors });

          // Create new data with RDA scores
          const rdaColumns = Array.from(
            { length: nComponents || model.constrainedAxes },
            (_, i) => `RDA${i + 1}`,
          );

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove response columns
            response.forEach((col) => delete newRow[col]);
            // Add RDA scores
            rdaColumns.forEach((rdaCol, j) => {
              newRow[rdaCol] = scores.siteScores[i][j];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer: { model, rdaColumns },
          };
        } else {
          // Transform using fitted RDA
          const scores = rda.transform(transformer.model, { data, response, predictors });

          const transformedData = data.map((row, i) => {
            const newRow = { ...row };
            // Remove response columns
            response.forEach((col) => delete newRow[col]);
            // Add RDA scores
            transformer.rdaColumns.forEach((rdaCol, j) => {
              newRow[rdaCol] = scores.siteScores[i][j];
            });
            return newRow;
          });

          return {
            data: transformedData,
            transformer,
          };
        }
      },
    });

    // Update feature list
    const rdaColumns = Array.from(
      { length: nComponents || Math.min(response.length, predictors.length) },
      (_, i) => `RDA${i + 1}`,
    );
    this.X = this.X.filter((col) => !response.includes(col)).concat(rdaColumns);

    return this;
  }

  /**
   * Split data into train/test sets
   * @param {Object} options - Split options
   * @param {number} options.ratio - Training ratio (default 0.7)
   * @param {boolean} options.shuffle - Shuffle before split
   * @param {number} options.seed - Random seed
   * @returns {Recipe} this
   */
  split({ ratio = 0.7, shuffle = true, seed = null } = {}) {
    this.splitConfig = { ratio, shuffle, seed };
    return this;
  }

  /**
   * Execute the recipe on the initial data
   * Returns train/test data and all fitted transformers
   * @returns {Object} Prepared data with train, test, transformers
   */
  prep() {
    let currentData = this.initialData;
    const transformers = {};
    const stepOutputs = [];

    // Apply preprocessing steps
    for (const step of this.steps) {
      if (step.type === 'encode' || step.type === 'scale' || step.type === 'dimreduction') {
        // These steps create transformers
        const result = step.fn(currentData, null);
        currentData = result.data;
        transformers[step.name] = result.transformer;
        step.transformer = result.transformer; // Store for later
        stepOutputs.push({
          name: step.name,
          output: currentData,
          transformer: result.transformer,
        });

        if (step.name === 'oneHot' && result.transformer instanceof Map) {
          const colsToRemove = new Set(Array.isArray(step.columns) ? step.columns : [step.columns]);
          const encodedColumns = [];
          for (const col of colsToRemove) {
            const info = result.transformer.get(col);
            if (info && Array.isArray(info.columnNames)) {
              encodedColumns.push(...info.columnNames);
            }
          }
          this.X = this.X
            .filter((col) => !colsToRemove.has(col))
            .concat(encodedColumns);
        }
      } else {
        // Simple transformation
        currentData = step.fn(currentData);
        stepOutputs.push({
          name: step.name,
          output: currentData,
        });
      }
    }

    // Split if configured
    let trainData, testData, splitResult;
    if (this.splitConfig) {
      splitResult = splitData(
        { data: currentData, X: this.X, y: this.y },
        this.splitConfig,
      );
      trainData = splitResult.train.data;
      testData = splitResult.test.data;
    } else {
      // No split - everything is "training"
      trainData = currentData;
      testData = null;
    }

    this._prepared = true;
    this._transformers = transformers;
    this._stepOutputs = stepOutputs;
    this._splitResult = splitResult;

    return {
      train: {
        data: trainData,
        X: this.X,
        y: this.y,
        metadata: splitResult?.train?.metadata,
      },
      test: testData
        ? {
          data: testData,
          X: this.X,
          y: this.y,
          metadata: splitResult?.test?.metadata,
        }
        : null,
      transformers,
      steps: stepOutputs,
      split: splitResult
        ? {
          train: splitResult.train,
          test: splitResult.test,
          ratio: this.splitConfig.ratio,
        }
        : null,
    };
  }

  /**
   * Apply fitted transformers to new data
   * @param {Array<Object>} data - New data to transform
   * @returns {Object} Transformed data
   */
  bake(data) {
    if (!this._prepared) {
      throw new Error('Recipe not prepared. Call prep() first.');
    }

    let currentData = data;
    const stepOutputs = [];

    // Apply preprocessing steps using fitted transformers
    for (const step of this.steps) {
      if (step.type === 'encode' || step.type === 'scale' || step.type === 'dimreduction') {
        // Use fitted transformer
        const result = step.fn(currentData, step.transformer);
        currentData = result.data;
        stepOutputs.push({
          name: step.name,
          output: currentData,
        });
      } else {
        // Simple transformation
        currentData = step.fn(currentData);
        stepOutputs.push({
          name: step.name,
          output: currentData,
        });
      }
    }

    return {
      data: currentData,
      X: this.X,
      y: this.y,
      steps: stepOutputs,
    };
  }

  /**
   * Get a summary of the recipe
   * @returns {string} Recipe summary
   */
  summary() {
    const lines = ['Recipe Summary:', ''];
    lines.push(`Initial data: ${this.X.length} features, target: ${this.y || 'none'}`);
    lines.push(`Steps: ${this.steps.length}`);
    lines.push('');

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      lines.push(`${i + 1}. ${step.name} (${step.type})`);
      if (step.columns) {
        lines.push(`   Columns: ${step.columns.join(', ')}`);
      }
    }

    if (this.splitConfig) {
      lines.push('');
      lines.push(
        `Split: ${this.splitConfig.ratio * 100}% train / ${
          (1 - this.splitConfig.ratio) * 100
        }% test`,
      );
    }

    return lines.join('\n');
  }
}
