/**
 * Model validation utilities
 * Cross-validation, train-test split, and other validation strategies
 */

import { setSeed, shuffle as shuffleUtil } from './utils.js';
import { prepareX, prepareXY } from '../core/table.js';

function isTableSplitInput(value) {
  return (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    (value.data || value.rows)
  );
}

function baseTrainTestSplit(X, y, { ratio = 0.8, shuffle = true, seed = null } = {}) {
  const n = X.length;

  if (n === 0) {
    throw new Error('trainTestSplit requires at least one sample');
  }

  if (seed !== null) {
    setSeed(seed);
  }

  let indices = Array.from({ length: n }, (_, i) => i);

  if (shuffle) {
    indices = shuffleUtil(indices);
  }

  const splitPoint = Math.floor(n * ratio);
  const trainIndices = indices.slice(0, splitPoint);
  const testIndices = indices.slice(splitPoint);

  const XTrain = trainIndices.map((i) => X[i]);
  const XTest = testIndices.map((i) => X[i]);

  if (y === null) {
    return { XTrain, XTest, trainIndices, testIndices };
  }

  const yTrain = trainIndices.map((i) => y[i]);
  const yTest = testIndices.map((i) => y[i]);

  return { XTrain, XTest, yTrain, yTest, trainIndices, testIndices };
}

function buildTableSplitView(dataset, baseSplit) {
  const { rows, columns, hasY, encoders } = dataset;
  const hasRows = Array.isArray(rows);

  const view = (indices, XSlice, ySlice) => {
    const rowSubset = hasRows ? indices.map((idx) => rows[idx]) : null;
    return {
      X: XSlice,
      y: ySlice,
      data: rowSubset,
      columns,
      indices,
      metadata: {
        columns,
        encoders,
        filteredRows: rows,
        totalRows: rows ? rows.length : null,
      },
    };
  };

  return {
    train: view(baseSplit.trainIndices, baseSplit.XTrain, hasY ? baseSplit.yTrain : null),
    test: view(baseSplit.testIndices, baseSplit.XTest, hasY ? baseSplit.yTest : null),
  };
}

function prepareDescriptorDataset(
  descriptor,
  { requireY = false, groupColumn = null, groupValues = null } = {},
) {
  const columns = descriptor.X || descriptor.columns;
  const data = descriptor.data || descriptor.rows;

  if (!data) {
    throw new Error('Descriptor inputs require a data property');
  }

  let prepared;
  let hasY = false;

  if (descriptor.y) {
    if (!columns) {
      throw new Error('Descriptor inputs with y must specify X/columns');
    }
    prepared = prepareXY({
      X: columns,
      y: descriptor.y,
      data,
      naOmit: descriptor.naOmit,
      omit_missing: descriptor.omit_missing,
      encode: descriptor.encode,
      encoders: descriptor.encoders,
    });
    hasY = true;
  } else {
    prepared = prepareX({
      columns,
      data,
      naOmit: descriptor.naOmit,
      omit_missing: descriptor.omit_missing,
      encode: descriptor.encode,
      encoders: descriptor.encoders,
    });
  }

  if (requireY && !hasY) {
    throw new Error('This operation requires a y column in the descriptor.');
  }

  let groupsResolved = null;
  let groupColumnName = groupColumn || descriptor.groupColumn || null;
  const descriptorGroups = descriptor.groups;

  if (Array.isArray(groupValues)) {
    groupsResolved = groupValues;
  } else if (Array.isArray(descriptorGroups)) {
    groupsResolved = descriptorGroups;
  } else if (typeof descriptorGroups === 'string') {
    groupColumnName = descriptorGroups;
  }

  if (!groupsResolved && groupColumnName) {
    groupsResolved = prepared.rows.map((row) => row[groupColumnName]);
  }

  if (groupsResolved && groupsResolved.length !== prepared.X.length) {
    throw new Error('groups array must match the number of rows after preprocessing');
  }

  const encoders = prepared.encoders || null;

  return {
    X: prepared.X,
    y: hasY ? prepared.y : null,
    columns: hasY ? prepared.columnsX : prepared.columns,
    rows: prepared.rows,
    hasY,
    groups: groupsResolved,
    encoders,
  };
}

/**
 * Split data into train and test sets.
 *
 * Array API:
 *   trainTestSplit(X, y?, { ratio=0.8, shuffle=true, seed=null })
 *     -> { XTrain, XTest, yTrain?, yTest?, trainIndices, testIndices }
 *
 * Declarative table API:
 *   trainTestSplit({ data, X, y }, options)
 *     -> above fields plus:
 *        { columns, rows, encoders, train: { data, X, y, indices, metadata }, test: {...} }
 *
 * The table views keep metadata (encoders, filtered rows) so downstream
 * preprocessors/estimators can reuse the same label encoding.
 */
function trainTestSplitFromTable(descriptor, options = {}) {
  const dataset = prepareDescriptorDataset(descriptor);
  const baseSplit = baseTrainTestSplit(dataset.X, dataset.hasY ? dataset.y : null, options);
  const tableViews = buildTableSplitView(dataset, baseSplit);

  return {
    ...baseSplit,
    columns: dataset.columns,
    rows: dataset.rows,
    encoders: dataset.encoders,
    ...tableViews,
  };
}

function resolveFoldOptions(argK, argShuffle, { defaultK = 5, defaultShuffle = false } = {}) {
  if (typeof argK === 'object' && argK !== null && !Array.isArray(argK)) {
    return {
      k: typeof argK.k === 'number' ? argK.k : defaultK,
      shuffle: Object.prototype.hasOwnProperty.call(argK, 'shuffle')
        ? Boolean(argK.shuffle)
        : defaultShuffle,
    };
  }
  return {
    k: typeof argK === 'number' ? argK : defaultK,
    shuffle: typeof argShuffle === 'boolean' ? argShuffle : defaultShuffle,
  };
}

function resolveKValue(arg, defaultK = 5) {
  if (typeof arg === 'object' && arg !== null && !Array.isArray(arg)) {
    return typeof arg.k === 'number' ? arg.k : defaultK;
  }
  return typeof arg === 'number' ? arg : defaultK;
}

function attachTableDataToFolds(folds, dataset) {
  const { rows, columns, X, y, hasY, encoders } = dataset;
  const hasRows = Array.isArray(rows);

  const makeView = (indices) => {
    const rowSubset = hasRows ? indices.map((i) => rows[i]) : null;
    return {
      X: indices.map((i) => X[i]),
      y: hasY ? indices.map((i) => y[i]) : null,
      data: rowSubset,
      columns,
      indices,
      metadata: {
        columns,
        encoders,
        filteredRows: rows,
        totalRows: rows ? rows.length : null,
      },
    };
  };

  return folds.map((fold) => ({
    ...fold,
    train: makeView(fold.trainIndices),
    test: makeView(fold.testIndices),
  }));
}

/**
 * Split data into train and test sets
 * Supports both raw matrices and declarative table descriptors
 */
export function trainTestSplit(X, y = null, options = {}) {
  if (isTableSplitInput(X)) {
    // When using table input, second argument is options
    return trainTestSplitFromTable(X, y || {});
  }
  return baseTrainTestSplit(X, y, options);
}

function baseKFold(n, k = 5, shuffle = false) {
  if (k < 2 || k > n) {
    throw new Error(`k must be between 2 and ${n}`);
  }

  let indices = Array.from({ length: n }, (_, i) => i);

  if (shuffle) {
    indices = shuffleUtil(indices);
  }

  const foldSize = Math.floor(n / k);
  const folds = [];

  for (let i = 0; i < k; i++) {
    const start = i * foldSize;
    const end = i === k - 1 ? n : (i + 1) * foldSize;

    const testIndices = indices.slice(start, end);
    const trainIndices = [...indices.slice(0, start), ...indices.slice(end)];

    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: k,
      description: 'KFold',
    });
  }

  return folds;
}

function baseStratifiedKFold(y, k = 5) {
  const n = y.length;

  if (k < 2 || k > n) {
    throw new Error(`k must be between 2 and ${n}`);
  }

  const classIndices = new Map();
  y.forEach((label, idx) => {
    if (!classIndices.has(label)) {
      classIndices.set(label, []);
    }
    classIndices.get(label).push(idx);
  });

  for (const [label, indices] of classIndices) {
    classIndices.set(label, shuffleUtil(indices));
  }

  const folds = Array.from({ length: k }, () => ({ train: [], test: [] }));

  for (const [, indices] of classIndices) {
    const classSize = indices.length;
    const foldSize = Math.floor(classSize / k);

    for (let i = 0; i < k; i++) {
      const start = i * foldSize;
      const end = i === k - 1 ? classSize : (i + 1) * foldSize;
      const testIndices = indices.slice(start, end);
      const trainIndices = [...indices.slice(0, start), ...indices.slice(end)];

      folds[i].test.push(...testIndices);
      folds[i].train.push(...trainIndices);
    }
  }

  return folds.map((fold, i) => ({
    trainIndices: fold.train,
    testIndices: fold.test,
    fold: i + 1,
    nSplits: k,
    description: 'StratifiedKFold',
  }));
}

function baseGroupKFold(groups, k = 5) {
  const n = groups.length;
  const uniqueGroups = [...new Set(groups)];
  const nGroups = uniqueGroups.length;

  if (k > nGroups) {
    throw new Error(`k (${k}) cannot be greater than number of groups (${nGroups})`);
  }

  const shuffledGroups = shuffleUtil(uniqueGroups);
  const baseSize = Math.floor(nGroups / k);
  const remainder = nGroups % k;
  const folds = [];

  let cursor = 0;
  for (let i = 0; i < k; i++) {
    const size = baseSize + (i < remainder ? 1 : 0);
    const end = i === k - 1 ? nGroups : cursor + size;
    const testGroups = new Set(shuffledGroups.slice(cursor, end));
    cursor = end;

    const trainIndices = [];
    const testIndices = [];

    groups.forEach((group, idx) => {
      if (testGroups.has(group)) {
        testIndices.push(idx);
      } else {
        trainIndices.push(idx);
      }
    });

    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: k,
      description: 'GroupKFold',
    });
  }

  return folds;
}

function baseLeaveOneOut(n) {
  const folds = [];
  for (let i = 0; i < n; i++) {
    const testIndices = [i];
    const trainIndices = [
      ...Array.from({ length: i }, (_, j) => j),
      ...Array.from({ length: n - i - 1 }, (_, j) => i + j + 1),
    ];
    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: n,
      description: 'LeaveOneOut',
    });
  }
  return folds;
}

function baseShuffleSplit(n, { nSplits = 5, testRatio = 0.2 } = {}) {
  const testSize = Math.max(1, Math.floor(n * testRatio));
  const splits = [];

  for (let i = 0; i < nSplits; i++) {
    const indices = shuffleUtil(Array.from({ length: n }, (_, j) => j));
    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);

    splits.push({
      trainIndices,
      testIndices,
      split: i + 1,
      nSplits,
      description: 'ShuffleSplit',
    });
  }

  return splits;
}

/**
 * K-Fold cross-validation generator
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {number} k - Number of folds
 * @param {boolean} shuffle - Whether to shuffle data
 * @returns {Array<Object>} Array of fold objects
 */
export function kFold(X, y, k = 5, shuffle = false) {
  if (isTableSplitInput(X)) {
    const dataset = prepareDescriptorDataset(X);
    const { k: resolvedK, shuffle: resolvedShuffle } = resolveFoldOptions(y, k);
    const folds = baseKFold(dataset.X.length, resolvedK, resolvedShuffle);
    return attachTableDataToFolds(folds, dataset);
  }

  if (!Array.isArray(X)) {
    throw new Error('kFold expects X to be an array');
  }
  if (Array.isArray(y) && y.length !== X.length) {
    throw new Error('X and y must have the same length');
  }

  return baseKFold(X.length, k, shuffle);
}

/**
 * Stratified K-Fold for classification with balanced class distribution
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target labels
 * @param {number} k - Number of folds
 * @returns {Array<Object>} Array of fold objects
 */
export function stratifiedKFold(X, y, k = 5) {
  if (isTableSplitInput(X)) {
    const dataset = prepareDescriptorDataset(X, { requireY: true });
    const resolvedK = resolveKValue(y, 5);
    const folds = baseStratifiedKFold(dataset.y, resolvedK);
    return attachTableDataToFolds(folds, dataset);
  }

  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error('stratifiedKFold expects array inputs for X and y');
  }

  if (X.length !== y.length) {
    throw new Error('X and y must have the same length');
  }

  return baseStratifiedKFold(y, k);
}

/**
 * Group K-Fold keeping group membership intact
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Array} groups - Group labels
 * @param {number} k - Number of folds
 * @returns {Array<Object>} Array of fold objects
 */
export function groupKFold(X, y, groups, k = 5) {
  if (isTableSplitInput(X)) {
    const options = y || {};
    const dataset = prepareDescriptorDataset(X, {
      groupColumn: typeof options.groups === 'string' ? options.groups : null,
      groupValues: Array.isArray(options.groups) ? options.groups : null,
    });
    const resolvedGroups = dataset.groups;
    if (!resolvedGroups) {
      throw new Error('groupKFold descriptor inputs require a groups column or array');
    }
    const resolvedK = resolveKValue(options, 5);
    const folds = baseGroupKFold(resolvedGroups, resolvedK);
    return attachTableDataToFolds(folds, dataset);
  }

  if (!Array.isArray(X)) {
    throw new Error('groupKFold expects array input for X');
  }

  if (!Array.isArray(groups) || groups.length !== X.length) {
    throw new Error('groups must have same length as X');
  }

  return baseGroupKFold(groups, k);
}

/**
 * Leave-One-Out cross-validation
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @returns {Array<Object>} Array of fold objects
 */
export function leaveOneOut(X, y) {
  if (isTableSplitInput(X)) {
    const dataset = prepareDescriptorDataset(X, { requireY: !!X.y });
    const folds = baseLeaveOneOut(dataset.X.length);
    return attachTableDataToFolds(folds, dataset);
  }

  if (!Array.isArray(X)) {
    throw new Error('leaveOneOut expects array input for X');
  }

  return baseLeaveOneOut(X.length);
}

/**
 * Shuffle Split - repeated random train-test splits
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Object} options - {nSplits, testRatio, seed}
 * @returns {Array<Object>} Array of split objects
 */
export function shuffleSplit(X, y, options = {}) {
  const isDescriptor = isTableSplitInput(X);
  const config = isDescriptor ? y || {} : options;
  const { nSplits = 5, testRatio = 0.2, seed = null } = config;

  if (seed !== null) {
    setSeed(seed);
  }

  if (isDescriptor) {
    const dataset = prepareDescriptorDataset(X, { requireY: false });
    const splits = baseShuffleSplit(dataset.X.length, { nSplits, testRatio });
    return attachTableDataToFolds(splits, dataset);
  }

  if (!Array.isArray(X)) {
    throw new Error('shuffleSplit expects array input for X');
  }

  return baseShuffleSplit(X.length, { nSplits, testRatio });
}

/**
 * Execute cross-validation with a model.
 *
 * Array API:
 *   crossValidate(fitFn, scoreFn, X, y, folds?)
 *
 * Declarative table API:
 *   crossValidate(fitFn, scoreFn, { data, X, y, encoders? }, options?)
 * Options can include { k, shuffle, folds } when using descriptors.
 *
 * Returns:
 *   { scores, meanScore, stdScore, nFolds, metadata?, tableFolds? }
 * When invoked with a descriptor, metadata/tableFolds include the training encoders
 * and per-fold table views for further inspection.
 */
export function crossValidate(fitFn, scoreFn, X, y = null, folds = null) {
  let dataX = X;
  let dataY = y;
  let foldDefs = folds;
  let dataset = null;
  let descriptorOptions = null;

  if (isTableSplitInput(X)) {
    if (Array.isArray(y)) {
      throw new Error('crossValidate: when using table descriptors, omit the separate y array.');
    }
    descriptorOptions = y && typeof y === 'object' ? y : {};
    dataset = prepareDescriptorDataset(X, { requireY: true });
    dataX = dataset.X;
    dataY = dataset.y;

    if (Array.isArray(folds) && folds.length) {
      foldDefs = folds;
    } else if (descriptorOptions && Array.isArray(descriptorOptions.folds)) {
      foldDefs = descriptorOptions.folds;
    } else {
      const resolvedK = typeof descriptorOptions?.folds === 'number'
        ? descriptorOptions.folds
        : typeof descriptorOptions?.k === 'number'
        ? descriptorOptions.k
        : 5;
      const resolvedShuffle = descriptorOptions?.shuffle ?? false;
      foldDefs = baseKFold(dataX.length, resolvedK, resolvedShuffle);
    }
  } else {
    if (!Array.isArray(dataX) || !Array.isArray(dataY)) {
      throw new Error('crossValidate expects array inputs for X and y');
    }
    if (!foldDefs) {
      foldDefs = baseKFold(dataX.length);
    }
  }

  if (!Array.isArray(foldDefs) || foldDefs.length === 0) {
    throw new Error('crossValidate requires at least one fold definition');
  }

  const scores = [];

  for (const fold of foldDefs) {
    const { trainIndices, testIndices } = fold;

    const XTrain = trainIndices.map((i) => dataX[i]);
    const yTrain = trainIndices.map((i) => dataY[i]);
    const XTest = testIndices.map((i) => dataX[i]);
    const yTest = testIndices.map((i) => dataY[i]);

    const model = fitFn(XTrain, yTrain);
    const score = scoreFn(model, XTest, yTest);
    scores.push(score);
  }

  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + (score - meanScore) ** 2, 0) / scores.length;
  const stdScore = Math.sqrt(variance);

  const result = {
    scores,
    meanScore,
    stdScore,
    nFolds: foldDefs.length,
  };

  if (dataset) {
    result.metadata = {
      columns: dataset.columns,
      encoders: dataset.encoders,
      totalRows: dataset.rows ? dataset.rows.length : null,
    };
    result.tableFolds = attachTableDataToFolds(foldDefs, dataset);
  }

  return result;
}
