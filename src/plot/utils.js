import { attachShow } from './show.js';
import { normalize } from '../core/table.js';

/**
 * Visualization utilities for model interpretation
 * Returns Observable Plot configuration objects
 */

/**
 * Normalize a colorBy (or labels) specification into a plain array of
 * per-observation values.
 *
 * Accepted forms:
 * - an array (used as-is)
 * - any iterable, e.g. an Arquero column or a typed array (converted)
 * - a { data, column } descriptor (column extracted from table-like data)
 * - a column-name string, resolved against the source rows the model kept
 *   from a declarative fit (e.g. pca.fit({ data, columns }) stores the
 *   naOmit-filtered rows so values stay aligned with the scores)
 *
 * @param {*} spec - colorBy specification
 * @param {Object|null} result - Fitted model (for string column lookup)
 * @param {string} name - Option name used in error messages
 * @returns {Array|null} Array of values, or null when spec is null
 */
export function resolveGroupValues(spec, result = null, name = 'colorBy') {
  if (spec === null || spec === undefined) return null;

  if (Array.isArray(spec)) return alignToKeptRows(spec, result);

  if (typeof spec === 'string') {
    const rows = result && result.rows;
    if (!rows) {
      throw new Error(
        `${name}: "${spec}" is a column name, but the model has no source rows. ` +
          `Column-name lookup requires fitting with table data (e.g. fit({ data, columns })); ` +
          `otherwise pass an array of values instead.`,
      );
    }
    if (rows.length > 0 && !(spec in rows[0])) {
      throw new Error(`${name}: column "${spec}" not found in the model's source data.`);
    }
    // `rows` are already the post-naOmit kept rows, so this is aligned.
    return rows.map((row) => row[spec]);
  }

  if (typeof spec === 'object' && spec.data !== undefined && spec.column !== undefined) {
    const rows = normalize(spec.data);
    return alignToKeptRows(rows.map((row) => row[spec.column]), result);
  }

  if (typeof spec[Symbol.iterator] === 'function') {
    // Arquero columns, typed arrays, Sets, generators...
    return alignToKeptRows(Array.from(spec), result);
  }

  throw new Error(
    `${name} must be an array, an iterable (e.g. an Arquero column), ` +
      `a { data, column } descriptor, or a column-name string.`,
  );
}

/**
 * Realign a full-length per-row array to the rows a model kept after naOmit.
 *
 * When `values` has one entry per ORIGINAL row (length === model.sourceLength)
 * but the model dropped rows with missing values during fitting, this selects
 * only the surviving rows (via model.rowIndices) so the result lines up
 * row-for-row with the ordination scores. Arrays that are already the kept
 * length (or models without naOmit metadata) are returned unchanged.
 *
 * @param {Array} values - Per-row values, one per original input row
 * @param {Object|null} result - Fitted model carrying { rowIndices, sourceLength }
 * @returns {Array}
 */
function alignToKeptRows(values, result) {
  if (
    result &&
    Array.isArray(result.rowIndices) &&
    typeof result.sourceLength === 'number' &&
    values.length === result.sourceLength &&
    result.rowIndices.length !== values.length
  ) {
    return result.rowIndices.map((i) => values[i]);
  }
  return values;
}

/**
 * Generate feature importance bar plot configuration
 * @param {Array<Object>} importances - Feature importance array
 * @param {Object} options - {width, height, topN}
 * @returns {Object} Plot configuration
 */
export function plotFeatureImportance(importances, { 
  width = 640, 
  height = 400,
  topN = 10
} = {}) {
  // Take top N features
  const topFeatures = importances.slice(0, topN);
  
  const data = topFeatures.map((imp, i) => ({
    feature: imp.feature?.toString() || `Feature ${imp.feature}`,
    importance: imp.importance,
    std: imp.std,
    rank: i + 1
  }));
  
  return attachShow({
    type: 'featureImportance',
    width,
    height,
    data: { features: data },
    axes: {
      x: { label: 'Importance' },
      y: { label: 'Feature' }
    },
    marks: [
      {
        type: 'barX',
        data: 'features',
        x: 'importance',
        y: 'feature',
        fill: 'steelblue',
        sort: { y: '-x' }
      }
    ]
  });
}

/**
 * Generate partial dependence plot configuration
 * @param {Object} pdResult - Result from partialDependence()
 * @param {Object} options - {width, height, featureName}
 * @returns {Object} Plot configuration
 */
export function plotPartialDependence(pdResult, { 
  width = 640, 
  height = 400,
  featureName = null
} = {}) {
  const { values, predictions, feature } = pdResult;
  
  const data = values.map((val, i) => ({
    value: val,
    prediction: predictions[i]
  }));
  
  const xLabel = featureName || `Feature ${feature}`;
  
  return attachShow({
    type: 'partialDependence',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: xLabel },
      y: { label: 'Partial Dependence', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'points',
        x: 'value',
        y: 'prediction',
        stroke: 'steelblue',
        strokeWidth: 2
      },
      {
        type: 'dot',
        data: 'points',
        x: 'value',
        y: 'prediction',
        fill: 'steelblue',
        r: 3
      }
    ]
  });
}

/**
 * Generate correlation matrix heatmap configuration
 * @param {Object} corrResult - Result from correlationMatrix()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotCorrelationMatrix(corrResult, { 
  width = 640, 
  height = 600
} = {}) {
  const { matrix, features } = corrResult;
  
  // Flatten matrix into data points
  const data = [];
  for (let i = 0; i < features.length; i++) {
    for (let j = 0; j < features.length; j++) {
      data.push({
        feature1: features[i],
        feature2: features[j],
        correlation: matrix[i][j],
        i,
        j
      });
    }
  }
  
  return attachShow({
    type: 'correlationMatrix',
    width,
    height,
    data: { cells: data },
    axes: {
      x: { label: null },
      y: { label: null }
    },
    marks: [
      {
        type: 'cell',
        data: 'cells',
        x: 'feature1',
        y: 'feature2',
        fill: 'correlation',
        fillScale: {
          domain: [-1, 1],
          scheme: 'RdBu',
          reverse: true
        }
      },
      {
        type: 'text',
        data: 'cells',
        x: 'feature1',
        y: 'feature2',
        text: d => d.correlation.toFixed(2),
        fontSize: 10
      }
    ]
  });
}

/**
 * Generate residual plot configuration
 * @param {Object} residualData - Result from residualPlotData()
 * @param {Object} options - {width, height, standardized}
 * @returns {Object} Plot configuration
 */
export function plotResiduals(residualData, { 
  width = 640, 
  height = 400,
  standardized = false
} = {}) {
  const { fitted, residuals, standardized: stdResiduals } = residualData;
  
  const yValues = standardized ? stdResiduals : residuals;
  const yLabel = standardized ? 'Standardized Residuals' : 'Residuals';
  
  const data = fitted.map((fit, i) => ({
    fitted: fit,
    residual: yValues[i]
  }));
  
  return attachShow({
    type: 'residuals',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: 'Fitted Values', grid: true },
      y: { label: yLabel, grid: true }
    },
    marks: [
      {
        type: 'dot',
        data: 'points',
        x: 'fitted',
        y: 'residual',
        fill: 'steelblue',
        r: 3,
        fillOpacity: 0.6
      },
      {
        type: 'ruleY',
        y: 0,
        stroke: 'red',
        strokeDasharray: '4,4'
      }
    ]
  });
}

/**
 * Generate Q-Q plot configuration for normality check
 * @param {Object} residualData - Result from residualPlotData()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotQQ(residualData, { 
  width = 400, 
  height = 400
} = {}) {
  const { standardized } = residualData;
  
  // Sort standardized residuals
  const sorted = [...standardized].sort((a, b) => a - b);
  const n = sorted.length;
  
  // Theoretical quantiles (standard normal)
  const theoreticalQuantiles = [];
  for (let i = 0; i < n; i++) {
    // Use approximate quantile for standard normal
    const p = (i + 0.5) / n;
    // Approximate inverse CDF of standard normal
    const q = Math.sqrt(2) * erfInv(2 * p - 1);
    theoreticalQuantiles.push(q);
  }
  
  const data = sorted.map((obs, i) => ({
    theoretical: theoreticalQuantiles[i],
    observed: obs
  }));
  
  // Add reference line
  const minVal = Math.min(...theoreticalQuantiles, ...sorted);
  const maxVal = Math.max(...theoreticalQuantiles, ...sorted);
  
  return attachShow({
    type: 'qq',
    width,
    height,
    data: { 
      points: data,
      reference: [
        { x: minVal, y: minVal },
        { x: maxVal, y: maxVal }
      ]
    },
    axes: {
      x: { label: 'Theoretical Quantiles', grid: true },
      y: { label: 'Sample Quantiles', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'reference',
        x: 'x',
        y: 'y',
        stroke: 'red',
        strokeDasharray: '4,4'
      },
      {
        type: 'dot',
        data: 'points',
        x: 'theoretical',
        y: 'observed',
        fill: 'steelblue',
        r: 3
      }
    ]
  });
}

/**
 * Approximate error function inverse (for Q-Q plot)
 * @param {number} x - Input value
 * @returns {number} Inverse erf
 */
function erfInv(x) {
  const a = 0.147;
  const b = 2 / (Math.PI * a) + Math.log(1 - x * x) / 2;
  const sqrt1 = Math.sqrt(b * b - Math.log(1 - x * x) / a);
  const sqrt2 = Math.sqrt(sqrt1 - b);
  return sqrt2 * Math.sign(x);
}

/**
 * Generate learning curve plot configuration
 * @param {Object} lcResult - Result from learningCurve()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotLearningCurve(lcResult, { 
  width = 640, 
  height = 400
} = {}) {
  const { trainSizes, trainScores, testScores } = lcResult;
  
  const data = trainSizes.map((size, i) => [
    { size, score: trainScores[i], type: 'train' },
    { size, score: testScores[i], type: 'test' }
  ]).flat();
  
  return attachShow({
    type: 'learningCurve',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: 'Training Set Size', grid: true },
      y: { label: 'Score', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'points',
        x: 'size',
        y: 'score',
        stroke: 'type',
        strokeWidth: 2
      },
      {
        type: 'dot',
        data: 'points',
        x: 'size',
        y: 'score',
        fill: 'type',
        r: 4
      }
    ],
    legend: {
      color: { domain: ['train', 'test'], label: 'Type' }
    }
  });
}
