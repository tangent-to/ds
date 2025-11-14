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
import { SimpleImputer, KNNImputer, IterativeImputer } from './impute.js';
import { IsolationForest, LocalOutlierFactor, MahalanobisDistance } from './outliers.js';
import * as pca from '../mva/pca.js';
import * as lda from '../mva/lda.js';
import * as rda from '../mva/rda.js';
import { mean, std } from '../core/math.js';

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
 *
 * Data Cleaning:
 * - parseNumeric(): Convert string columns to numbers
 * - clean(): Remove rows with invalid categorical values
 *
 * Missing Value Imputation:
 * - imputeMean(): Impute with mean values
 * - imputeMedian(): Impute with median values
 * - imputeMode(): Impute with mode (most frequent)
 * - imputeKNN(): Impute using K-Nearest Neighbors
 * - imputeIterative(): Impute using iterative MICE algorithm
 *
 * Outlier Handling:
 * - removeOutliers(): Remove outliers using isolation forest, LOF, or Mahalanobis distance
 * - clipOutliers(): Clip outliers using IQR method
 *
 * Encoding:
 * - oneHot(): One-hot encode categorical columns
 *
 * Scaling:
 * - scale(): Scale numeric columns (standard or minmax)
 *
 * Feature Engineering:
 * - createInteractions(): Create pairwise interaction features
 * - createPolynomial(): Create polynomial features
 * - binContinuous(): Bin continuous variables into discrete categories
 *
 * Dimensionality Reduction:
 * - pca(): Principal Component Analysis
 * - lda(): Linear Discriminant Analysis (supervised)
 * - rda(): Redundancy Analysis (constrained ordination)
 *
 * Sampling:
 * - upsample(): Oversample minority class for imbalanced data
 * - downsample(): Undersample majority class for imbalanced data
 *
 * Feature Selection:
 * - selectByVariance(): Remove low-variance features
 * - selectByCorrelation(): Remove highly correlated features
 *
 * Data Splitting:
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
   * Impute missing values with mean
   * @param {Array<string>} columns - Columns to impute
   * @returns {Recipe} this
   *
   * @example
   * recipe.imputeMean(['age', 'income']);
   */
  imputeMean(columns) {
    this.steps.push({
      name: 'imputeMean',
      type: 'impute',
      columns,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit imputer
          const imputer = new SimpleImputer({ strategy: 'mean' });
          imputer.fit({ data, columns });
          return {
            data: imputer.transform({ data, columns }).data,
            transformer: imputer,
          };
        } else {
          // Transform using fitted imputer
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
   * Impute missing values with median
   * @param {Array<string>} columns - Columns to impute
   * @returns {Recipe} this
   *
   * @example
   * recipe.imputeMedian(['age', 'price']);
   */
  imputeMedian(columns) {
    this.steps.push({
      name: 'imputeMedian',
      type: 'impute',
      columns,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit imputer
          const imputer = new SimpleImputer({ strategy: 'median' });
          imputer.fit({ data, columns });
          return {
            data: imputer.transform({ data, columns }).data,
            transformer: imputer,
          };
        } else {
          // Transform using fitted imputer
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
   * Impute missing values with mode (most frequent value)
   * @param {Array<string>} columns - Columns to impute
   * @returns {Recipe} this
   *
   * @example
   * recipe.imputeMode(['category', 'status']);
   */
  imputeMode(columns) {
    this.steps.push({
      name: 'imputeMode',
      type: 'impute',
      columns,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit imputer
          const imputer = new SimpleImputer({ strategy: 'most_frequent' });
          imputer.fit({ data, columns });
          return {
            data: imputer.transform({ data, columns }).data,
            transformer: imputer,
          };
        } else {
          // Transform using fitted imputer
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
   * Impute missing values using KNN
   * @param {Array<string>} columns - Columns to impute
   * @param {Object} options - KNN imputer options
   * @param {number} options.k - Number of neighbors (default 5)
   * @returns {Recipe} this
   *
   * @example
   * recipe.imputeKNN(['age', 'income'], { k: 3 });
   */
  imputeKNN(columns, { k = 5 } = {}) {
    this.steps.push({
      name: 'imputeKNN',
      type: 'impute',
      columns,
      k,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit imputer
          const imputer = new KNNImputer({ k });
          imputer.fit({ data, columns });
          return {
            data: imputer.transform({ data, columns }).data,
            transformer: imputer,
          };
        } else {
          // Transform using fitted imputer
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
   * Impute missing values using iterative imputation (MICE)
   * @param {Array<string>} columns - Columns to impute
   * @param {Object} options - Iterative imputer options
   * @param {number} options.maxIter - Maximum iterations (default 10)
   * @param {number} options.tol - Convergence tolerance (default 0.001)
   * @returns {Recipe} this
   *
   * @example
   * recipe.imputeIterative(['age', 'income'], { maxIter: 20 });
   */
  imputeIterative(columns, { maxIter = 10, tol = 0.001 } = {}) {
    this.steps.push({
      name: 'imputeIterative',
      type: 'impute',
      columns,
      maxIter,
      tol,
      transformer: null,
      fn: (data, transformer) => {
        if (!transformer) {
          // Fit imputer
          const imputer = new IterativeImputer({ maxIter, tol });
          imputer.fit({ data, columns });
          return {
            data: imputer.transform({ data, columns }).data,
            transformer: imputer,
          };
        } else {
          // Transform using fitted imputer
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
   * Remove outliers from the dataset
   * @param {Array<string>} columns - Columns to check for outliers
   * @param {Object} options - Outlier detection options
   * @param {string} options.method - Detection method: 'isolation_forest', 'lof', or 'mahalanobis'
   * @param {number} options.contamination - Expected proportion of outliers (default 0.1)
   * @returns {Recipe} this
   *
   * @example
   * recipe.removeOutliers(['price', 'quantity'], { method: 'isolation_forest', contamination: 0.05 });
   */
  removeOutliers(columns, { method = 'isolation_forest', contamination = 0.1 } = {}) {
    this.steps.push({
      name: 'removeOutliers',
      type: 'filter',
      columns,
      method,
      contamination,
      fn: (data) => {
        let detector;
        if (method === 'isolation_forest') {
          detector = new IsolationForest({ contamination });
        } else if (method === 'lof') {
          detector = new LocalOutlierFactor({ contamination });
        } else if (method === 'mahalanobis') {
          detector = new MahalanobisDistance({ contamination });
        } else {
          throw new Error(`Unknown outlier detection method: ${method}`);
        }

        detector.fit({ data, columns });
        const predictions = detector.predict({ data, columns });

        // Keep only inliers (prediction === 1)
        const rows = table.normalize(data);
        return rows.filter((_, i) => predictions[i] === 1);
      },
    });
    return this;
  }

  /**
   * Clip outliers using IQR method
   * @param {Array<string>} columns - Columns to clip
   * @param {Object} options - Clipping options
   * @param {number} options.multiplier - IQR multiplier (default 1.5)
   * @returns {Recipe} this
   *
   * @example
   * recipe.clipOutliers(['price', 'age'], { multiplier: 1.5 });
   */
  clipOutliers(columns, { multiplier = 1.5 } = {}) {
    this.steps.push({
      name: 'clipOutliers',
      type: 'transform',
      columns,
      multiplier,
      transformer: null,
      fn: (data, transformer) => {
        const rows = table.normalize(data);

        if (!transformer) {
          // Fit: calculate bounds for each column
          const bounds = {};
          for (const col of columns) {
            const values = rows
              .map((row) => row[col])
              .filter((v) => v !== null && v !== undefined && !isNaN(v))
              .map(Number)
              .sort((a, b) => a - b);

            if (values.length === 0) {
              bounds[col] = { lower: -Infinity, upper: Infinity };
              continue;
            }

            const q1Index = Math.floor(values.length * 0.25);
            const q3Index = Math.floor(values.length * 0.75);
            const q1 = values[q1Index];
            const q3 = values[q3Index];
            const iqr = q3 - q1;

            bounds[col] = {
              lower: q1 - multiplier * iqr,
              upper: q3 + multiplier * iqr,
            };
          }

          // Transform: clip values
          const clipped = rows.map((row) => {
            const newRow = { ...row };
            for (const col of columns) {
              const val = row[col];
              if (val !== null && val !== undefined && !isNaN(val)) {
                const numVal = Number(val);
                const { lower, upper } = bounds[col];
                newRow[col] = Math.max(lower, Math.min(upper, numVal));
              }
            }
            return newRow;
          });

          return {
            data: clipped,
            transformer: bounds,
          };
        } else {
          // Transform using fitted bounds
          const clipped = rows.map((row) => {
            const newRow = { ...row };
            for (const col of columns) {
              const val = row[col];
              if (val !== null && val !== undefined && !isNaN(val)) {
                const numVal = Number(val);
                const { lower, upper } = transformer[col];
                newRow[col] = Math.max(lower, Math.min(upper, numVal));
              }
            }
            return newRow;
          });

          return {
            data: clipped,
            transformer,
          };
        }
      },
    });
    return this;
  }

  /**
   * Create pairwise interaction features
   * @param {Array<string>} columns - Columns to create interactions from
   * @returns {Recipe} this
   *
   * @example
   * recipe.createInteractions(['feature1', 'feature2', 'feature3']);
   * // Creates: feature1_x_feature2, feature1_x_feature3, feature2_x_feature3
   */
  createInteractions(columns) {
    this.steps.push({
      name: 'createInteractions',
      type: 'transform',
      columns,
      fn: (data) => {
        const rows = table.normalize(data);
        const newColumns = [];

        return rows.map((row) => {
          const newRow = { ...row };

          for (let i = 0; i < columns.length; i++) {
            for (let j = i + 1; j < columns.length; j++) {
              const col1 = columns[i];
              const col2 = columns[j];
              const interactionCol = `${col1}_x_${col2}`;

              const val1 = row[col1];
              const val2 = row[col2];

              if (val1 !== null && val1 !== undefined && val2 !== null && val2 !== undefined) {
                newRow[interactionCol] = Number(val1) * Number(val2);
                if (i === 0 && j === 1) {
                  newColumns.push(interactionCol);
                }
              } else {
                newRow[interactionCol] = null;
              }
            }
          }

          return newRow;
        });
      },
    });

    // Update feature list with new interaction columns
    const newColumns = [];
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        newColumns.push(`${columns[i]}_x_${columns[j]}`);
      }
    }
    this.X = this.X.concat(newColumns);

    return this;
  }

  /**
   * Create polynomial features
   * @param {Array<string>} columns - Columns to create polynomials from
   * @param {Object} options - Polynomial options
   * @param {number} options.degree - Polynomial degree (default 2)
   * @returns {Recipe} this
   *
   * @example
   * recipe.createPolynomial(['age', 'income'], { degree: 2 });
   * // Creates: age^2, income^2
   */
  createPolynomial(columns, { degree = 2 } = {}) {
    this.steps.push({
      name: 'createPolynomial',
      type: 'transform',
      columns,
      degree,
      fn: (data) => {
        const rows = table.normalize(data);

        return rows.map((row) => {
          const newRow = { ...row };

          for (const col of columns) {
            const val = row[col];
            if (val !== null && val !== undefined && !isNaN(val)) {
              const numVal = Number(val);
              for (let d = 2; d <= degree; d++) {
                newRow[`${col}^${d}`] = Math.pow(numVal, d);
              }
            } else {
              for (let d = 2; d <= degree; d++) {
                newRow[`${col}^${d}`] = null;
              }
            }
          }

          return newRow;
        });
      },
    });

    // Update feature list with polynomial columns
    const newColumns = [];
    for (const col of columns) {
      for (let d = 2; d <= degree; d++) {
        newColumns.push(`${col}^${d}`);
      }
    }
    this.X = this.X.concat(newColumns);

    return this;
  }

  /**
   * Bin continuous variables into discrete categories
   * @param {string} column - Column to bin
   * @param {Object} options - Binning options
   * @param {number} options.bins - Number of bins (default 5)
   * @param {Array<string>} options.labels - Custom bin labels
   * @returns {Recipe} this
   *
   * @example
   * recipe.binContinuous('age', { bins: 5, labels: ['child', 'teen', 'adult', 'middle', 'senior'] });
   */
  binContinuous(column, { bins = 5, labels = null } = {}) {
    this.steps.push({
      name: 'binContinuous',
      type: 'transform',
      column,
      bins,
      labels,
      transformer: null,
      fn: (data, transformer) => {
        const rows = table.normalize(data);

        if (!transformer) {
          // Fit: calculate bin edges
          const values = rows
            .map((row) => row[column])
            .filter((v) => v !== null && v !== undefined && !isNaN(v))
            .map(Number)
            .sort((a, b) => a - b);

          if (values.length === 0) {
            return {
              data: rows,
              transformer: { edges: [], labels: [] },
            };
          }

          const min = values[0];
          const max = values[values.length - 1];
          const step = (max - min) / bins;

          const edges = [];
          for (let i = 0; i <= bins; i++) {
            edges.push(min + i * step);
          }

          const binLabels = labels || Array.from({ length: bins }, (_, i) => `bin_${i + 1}`);

          // Transform: assign bins
          const binned = rows.map((row) => {
            const newRow = { ...row };
            const val = row[column];

            if (val === null || val === undefined || isNaN(val)) {
              newRow[column] = null;
            } else {
              const numVal = Number(val);
              let binIndex = 0;

              for (let i = 0; i < edges.length - 1; i++) {
                if (numVal >= edges[i] && (numVal < edges[i + 1] || i === edges.length - 2)) {
                  binIndex = i;
                  break;
                }
              }

              newRow[column] = binLabels[binIndex];
            }

            return newRow;
          });

          return {
            data: binned,
            transformer: { edges, labels: binLabels },
          };
        } else {
          // Transform using fitted edges
          const { edges, labels: binLabels } = transformer;

          const binned = rows.map((row) => {
            const newRow = { ...row };
            const val = row[column];

            if (val === null || val === undefined || isNaN(val)) {
              newRow[column] = null;
            } else {
              const numVal = Number(val);
              let binIndex = 0;

              for (let i = 0; i < edges.length - 1; i++) {
                if (numVal >= edges[i] && (numVal < edges[i + 1] || i === edges.length - 2)) {
                  binIndex = i;
                  break;
                }
              }

              newRow[column] = binLabels[binIndex];
            }

            return newRow;
          });

          return {
            data: binned,
            transformer,
          };
        }
      },
    });

    return this;
  }

  /**
   * Upsample minority class for imbalanced classification
   * @param {Object} options - Upsampling options
   * @param {number} options.targetRatio - Target ratio of minority to majority (default 1.0 for balanced)
   * @param {number} options.seed - Random seed
   * @returns {Recipe} this
   *
   * @example
   * recipe.upsample({ targetRatio: 1.0, seed: 42 });
   */
  upsample({ targetRatio = 1.0, seed = null } = {}) {
    if (!this.y) {
      throw new Error('Upsampling requires a target variable (y). Specify y in recipe().');
    }

    this.steps.push({
      name: 'upsample',
      type: 'transform',
      targetRatio,
      seed,
      fn: (data) => {
        const rows = table.normalize(data);
        const rng = seed !== null ? () => {
          const x = Math.sin(seed++) * 10000;
          return x - Math.floor(x);
        } : Math.random;

        // Count class frequencies
        const classCounts = new Map();
        rows.forEach((row) => {
          const label = row[this.y];
          classCounts.set(label, (classCounts.get(label) || 0) + 1);
        });

        // Find majority and minority classes
        const counts = Array.from(classCounts.entries());
        counts.sort((a, b) => b[1] - a[1]);
        const majorityCount = counts[0][1];

        // Upsample each minority class
        const upsampled = [...rows];

        for (const [label, count] of counts) {
          if (count < majorityCount) {
            const targetCount = Math.floor(majorityCount * targetRatio);
            const classRows = rows.filter((row) => row[this.y] === label);
            const samplesToAdd = targetCount - count;

            for (let i = 0; i < samplesToAdd; i++) {
              const randomRow = classRows[Math.floor(rng() * classRows.length)];
              upsampled.push({ ...randomRow });
            }
          }
        }

        return upsampled;
      },
    });

    return this;
  }

  /**
   * Downsample majority class for imbalanced classification
   * @param {Object} options - Downsampling options
   * @param {string} options.strategy - 'balance' (equal classes) or 'ratio' (custom ratio)
   * @param {number} options.targetRatio - Target ratio of majority to minority (for 'ratio' strategy)
   * @param {number} options.seed - Random seed
   * @returns {Recipe} this
   *
   * @example
   * recipe.downsample({ strategy: 'balance', seed: 42 });
   */
  downsample({ strategy = 'balance', targetRatio = 1.0, seed = null } = {}) {
    if (!this.y) {
      throw new Error('Downsampling requires a target variable (y). Specify y in recipe().');
    }

    this.steps.push({
      name: 'downsample',
      type: 'transform',
      strategy,
      targetRatio,
      seed,
      fn: (data) => {
        const rows = table.normalize(data);
        const rng = seed !== null ? () => {
          const x = Math.sin(seed++) * 10000;
          return x - Math.floor(x);
        } : Math.random;

        // Count class frequencies
        const classCounts = new Map();
        const classSamples = new Map();
        rows.forEach((row) => {
          const label = row[this.y];
          classCounts.set(label, (classCounts.get(label) || 0) + 1);
          if (!classSamples.has(label)) {
            classSamples.set(label, []);
          }
          classSamples.get(label).push(row);
        });

        // Find majority and minority classes
        const counts = Array.from(classCounts.entries());
        counts.sort((a, b) => a[1] - b[1]);
        const minorityCount = counts[0][1];

        // Downsample majority classes
        const downsampled = [];

        for (const [label, count] of classCounts.entries()) {
          const samples = classSamples.get(label);

          if (strategy === 'balance') {
            if (count > minorityCount) {
              // Randomly sample minorityCount samples
              const shuffled = [...samples].sort(() => rng() - 0.5);
              downsampled.push(...shuffled.slice(0, minorityCount));
            } else {
              downsampled.push(...samples);
            }
          } else if (strategy === 'ratio') {
            const targetCount = Math.floor(minorityCount * targetRatio);
            if (count > targetCount) {
              const shuffled = [...samples].sort(() => rng() - 0.5);
              downsampled.push(...shuffled.slice(0, targetCount));
            } else {
              downsampled.push(...samples);
            }
          }
        }

        return downsampled;
      },
    });

    return this;
  }

  /**
   * Remove low-variance features
   * @param {Object} options - Feature selection options
   * @param {number} options.threshold - Variance threshold (default 0.0)
   * @returns {Recipe} this
   *
   * @example
   * recipe.selectByVariance({ threshold: 0.01 });
   */
  selectByVariance({ threshold = 0.0 } = {}) {
    this.steps.push({
      name: 'selectByVariance',
      type: 'transform',
      threshold,
      transformer: null,
      fn: (data, transformer) => {
        const rows = table.normalize(data);

        if (!transformer) {
          // Fit: calculate variance for each numeric column
          const variances = {};
          const columnsToKeep = [];

          for (const col of this.X) {
            const values = rows
              .map((row) => row[col])
              .filter((v) => v !== null && v !== undefined && !isNaN(v))
              .map(Number);

            if (values.length === 0) {
              columnsToKeep.push(col);
              continue;
            }

            const meanVal = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((sum, v) => sum + Math.pow(v - meanVal, 2), 0) / values.length;

            variances[col] = variance;

            if (variance > threshold) {
              columnsToKeep.push(col);
            }
          }

          // Transform: keep only high-variance columns
          const filtered = rows.map((row) => {
            const newRow = {};
            for (const col of columnsToKeep) {
              newRow[col] = row[col];
            }
            if (this.y) {
              newRow[this.y] = row[this.y];
            }
            return newRow;
          });

          return {
            data: filtered,
            transformer: { columnsToKeep, variances },
          };
        } else {
          // Transform: keep only selected columns
          const { columnsToKeep } = transformer;
          const filtered = rows.map((row) => {
            const newRow = {};
            for (const col of columnsToKeep) {
              newRow[col] = row[col];
            }
            if (this.y) {
              newRow[this.y] = row[this.y];
            }
            return newRow;
          });

          return {
            data: filtered,
            transformer,
          };
        }
      },
    });

    return this;
  }

  /**
   * Remove highly correlated features
   * @param {Object} options - Feature selection options
   * @param {number} options.threshold - Correlation threshold (default 0.95)
   * @returns {Recipe} this
   *
   * @example
   * recipe.selectByCorrelation({ threshold: 0.9 });
   */
  selectByCorrelation({ threshold = 0.95 } = {}) {
    this.steps.push({
      name: 'selectByCorrelation',
      type: 'transform',
      threshold,
      transformer: null,
      fn: (data, transformer) => {
        const rows = table.normalize(data);

        if (!transformer) {
          // Fit: calculate correlations and select columns to keep
          const numericCols = this.X.filter((col) => {
            const values = rows.map((row) => row[col]);
            return values.some((v) => typeof v === 'number');
          });

          // Calculate correlation matrix
          const correlations = {};
          for (let i = 0; i < numericCols.length; i++) {
            for (let j = i + 1; j < numericCols.length; j++) {
              const col1 = numericCols[i];
              const col2 = numericCols[j];

              const values1 = rows.map((row) => Number(row[col1])).filter((v) => !isNaN(v));
              const values2 = rows.map((row) => Number(row[col2])).filter((v) => !isNaN(v));

              if (values1.length === 0 || values2.length === 0) continue;

              const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
              const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
              const std1 = Math.sqrt(
                values1.reduce((sum, v) => sum + Math.pow(v - mean1, 2), 0) / values1.length,
              );
              const std2 = Math.sqrt(
                values2.reduce((sum, v) => sum + Math.pow(v - mean2, 2), 0) / values2.length,
              );

              if (std1 === 0 || std2 === 0) continue;

              let covariance = 0;
              for (let k = 0; k < Math.min(values1.length, values2.length); k++) {
                covariance += (values1[k] - mean1) * (values2[k] - mean2);
              }
              covariance /= Math.min(values1.length, values2.length);

              const correlation = covariance / (std1 * std2);
              correlations[`${col1}_${col2}`] = Math.abs(correlation);
            }
          }

          // Remove columns with high correlation
          const columnsToRemove = new Set();
          for (const [pair, corr] of Object.entries(correlations)) {
            if (corr > threshold) {
              const [col1, col2] = pair.split('_');
              // Remove col2 (arbitrary choice)
              columnsToRemove.add(col2);
            }
          }

          const columnsToKeep = this.X.filter((col) => !columnsToRemove.has(col));

          // Transform: keep only selected columns
          const filtered = rows.map((row) => {
            const newRow = {};
            for (const col of columnsToKeep) {
              newRow[col] = row[col];
            }
            if (this.y) {
              newRow[this.y] = row[this.y];
            }
            return newRow;
          });

          return {
            data: filtered,
            transformer: { columnsToKeep, correlations },
          };
        } else {
          // Transform: keep only selected columns
          const { columnsToKeep } = transformer;
          const filtered = rows.map((row) => {
            const newRow = {};
            for (const col of columnsToKeep) {
              newRow[col] = row[col];
            }
            if (this.y) {
              newRow[this.y] = row[this.y];
            }
            return newRow;
          });

          return {
            data: filtered,
            transformer,
          };
        }
      },
    });

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
      // Check if this step type creates transformers
      const createsTransformer =
        step.type === 'encode' ||
        step.type === 'scale' ||
        step.type === 'dimreduction' ||
        step.type === 'impute' ||
        (step.type === 'transform' && step.transformer !== undefined);

      if (createsTransformer) {
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

        // Handle column updates for oneHot encoding
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

        // Handle column updates for feature selection
        if ((step.name === 'selectByVariance' || step.name === 'selectByCorrelation') && result.transformer) {
          this.X = result.transformer.columnsToKeep;
        }
      } else {
        // Simple transformation without transformer
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
      // Check if this step type uses transformers
      const usesTransformer =
        step.type === 'encode' ||
        step.type === 'scale' ||
        step.type === 'dimreduction' ||
        step.type === 'impute' ||
        (step.type === 'transform' && step.transformer !== undefined);

      if (usesTransformer) {
        // Use fitted transformer
        const result = step.fn(currentData, step.transformer);
        currentData = result.data;
        stepOutputs.push({
          name: step.name,
          output: currentData,
        });
      } else {
        // Simple transformation without transformer
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
