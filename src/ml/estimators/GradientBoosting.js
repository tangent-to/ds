/**
 * Gradient Boosting estimators built on top of Decision Trees.
 *
 * Classic Friedman-style gradient boosting machines:
 * - GradientBoostingRegressor: squared-error loss, trees fit to residuals
 * - GradientBoostingClassifier: logistic deviance (binary) or multinomial
 *   deviance (multiclass) with Newton leaf-value updates per tree
 */

import { Classifier, Estimator, Regressor } from "../../core/estimators/estimator.js";
import { prepareX, prepareXY } from "../../core/table.js";
import { DecisionTreeRegressor } from "./DecisionTree.js";

const EPS = 1e-10;

function toNumericMatrix(X) {
  return X.map((row) => (Array.isArray(row) ? row.map(Number) : [Number(row)]));
}

function createRandomGenerator(seed) {
  if (seed === null || seed === undefined) {
    return Math.random;
  }
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function prepareDataset(X, y) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
      encoders: X.encoders,
    });
    return {
      X: toNumericMatrix(prepared.X),
      y: prepared.y,
      columns: prepared.columnsX,
      encoders: prepared.encoders,
    };
  }
  return {
    X: toNumericMatrix(X),
    y: Array.isArray(y) ? y.slice() : Array.from(y),
    columns: null,
    encoders: null,
  };
}

function preparePredict(X, columns) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || columns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

/**
 * Sample `fraction` of indices 0..n-1 without replacement (partial Fisher-Yates)
 */
function subsampleIndices(n, fraction, random) {
  const size = Math.max(1, Math.floor(n * fraction));
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < size; i++) {
    const j = i + Math.floor(random() * (n - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, size);
}

function findLeaf(node, row) {
  while (node.type !== "leaf") {
    node = row[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node;
}

function sigmoid(v) {
  return 1 / (1 + Math.exp(-Math.max(-700, Math.min(700, v))));
}

function softmaxRow(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

class GradientBoostingBase extends Estimator {
  constructor({
    nEstimators = 100,
    learningRate = 0.1,
    maxDepth = 3,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    minGain = 0.0,
    subsample = 1.0,
    maxFeatures = null,
    seed = null,
    task = "regression",
  } = {}) {
    super({
      nEstimators,
      learningRate,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      minGain,
      subsample,
      maxFeatures,
      seed,
      task,
    });
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minGain = minGain;
    this.subsample = subsample;
    this.maxFeatures = maxFeatures;
    this.seed = seed;
    this.task = task;
    this.random = createRandomGenerator(seed);
    this.trees = []; // regression/binary: trees; multiclass: arrays of K trees
    this.init_ = null;
    this.nClasses_ = null;
    this.columns = null;
    this.lossHistory_ = [];
    this._featureImportances = null;
  }

  _newTree() {
    return new DecisionTreeRegressor({
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      minGain: this.minGain,
      maxFeatures: this.maxFeatures,
      random: this.random,
    });
  }

  /**
   * Fit one tree to the negative gradient on a subsample and return its
   * predictions for all rows. `leafUpdate(leafSamples) -> value` optionally
   * replaces each leaf value with a Newton step computed from the subsample
   * rows that fall into that leaf.
   */
  _fitStageTree(X, negGradient, sampleIdx, leafUpdate = null) {
    const Xs = sampleIdx.map((i) => X[i]);
    const rs = sampleIdx.map((i) => negGradient[i]);

    const tree = this._newTree();
    tree.fit(Xs, rs);

    if (leafUpdate) {
      const root = tree.tree.root;
      const leafSamples = new Map();
      for (const i of sampleIdx) {
        const leaf = findLeaf(root, X[i]);
        if (!leafSamples.has(leaf)) leafSamples.set(leaf, []);
        leafSamples.get(leaf).push(i);
      }
      for (const [leaf, samples] of leafSamples) {
        leaf.value = leafUpdate(samples);
      }
    }

    return { tree, update: X.map((row) => findLeaf(tree.tree.root, row).value) };
  }

  /**
   * Fit the gradient boosting model on training data.
   * @param {Array<Array<number>>} X - Feature matrix (n samples × p features)
   * @param {Array<number>} y - Target values (class indices or regression targets)
   * @returns {this} The fitted estimator (for chaining)
   */
  fit(X, y) {
    const prepared = prepareDataset(X, y);
    return this._fitPrepared(prepared.X, prepared.y, prepared.columns);
  }

  _fitPrepared(X, y, columns) {
    this.columns = columns;
    this.trees = [];
    this.lossHistory_ = [];

    if (this.task === "regression") {
      this._fitRegression(X, y);
    } else {
      this._fitClassification(X, y);
    }

    this._computeFeatureImportances(X[0].length);
    this.fitted = true;
    return this;
  }

  _fitRegression(X, y) {
    const n = X.length;
    this.init_ = y.reduce((a, b) => a + b, 0) / n;
    const F = new Array(n).fill(this.init_);

    for (let m = 0; m < this.nEstimators; m++) {
      const residuals = y.map((yi, i) => yi - F[i]);
      const idx = this.subsample < 1
        ? subsampleIndices(n, this.subsample, this.random)
        : residuals.map((_, i) => i);

      const { tree, update } = this._fitStageTree(X, residuals, idx);
      this.trees.push(tree);

      for (let i = 0; i < n; i++) {
        F[i] += this.learningRate * update[i];
      }

      const mse = y.reduce((s, yi, i) => s + (yi - F[i]) ** 2, 0) / n;
      this.lossHistory_.push(mse);
    }
  }

  /**
   * y must be class indices 0..K-1
   */
  _fitClassification(X, y) {
    const n = X.length;
    const K = Math.max(...y) + 1;
    this.nClasses_ = K;

    if (K === 2) {
      this._fitBinary(X, y, n);
    } else {
      this._fitMulticlass(X, y, n, K);
    }
  }

  _fitBinary(X, y, n) {
    const p0 = Math.min(
      1 - EPS,
      Math.max(EPS, y.reduce((a, b) => a + b, 0) / n),
    );
    this.init_ = Math.log(p0 / (1 - p0));
    const F = new Array(n).fill(this.init_);

    for (let m = 0; m < this.nEstimators; m++) {
      const probs = F.map(sigmoid);
      const residuals = y.map((yi, i) => yi - probs[i]);
      const idx = this.subsample < 1
        ? subsampleIndices(n, this.subsample, this.random)
        : residuals.map((_, i) => i);

      // Newton step per leaf: gamma = sum(r) / sum(p * (1 - p))
      const { tree, update } = this._fitStageTree(X, residuals, idx, (samples) => {
        let num = 0;
        let den = 0;
        for (const i of samples) {
          num += residuals[i];
          den += probs[i] * (1 - probs[i]);
        }
        return num / Math.max(den, EPS);
      });
      this.trees.push(tree);

      for (let i = 0; i < n; i++) {
        F[i] += this.learningRate * update[i];
      }

      const deviance = -y.reduce((s, yi, i) => {
        const p = Math.min(1 - EPS, Math.max(EPS, sigmoid(F[i])));
        return s + yi * Math.log(p) + (1 - yi) * Math.log(1 - p);
      }, 0) / n;
      this.lossHistory_.push(deviance);
    }
  }

  _fitMulticlass(X, y, n, K) {
    const priors = new Array(K).fill(0);
    for (const yi of y) priors[yi]++;
    this.init_ = priors.map((c) => Math.log(Math.max(c / n, EPS)));

    const F = Array.from({ length: n }, () => this.init_.slice());

    for (let m = 0; m < this.nEstimators; m++) {
      const P = F.map(softmaxRow);
      // One subsample per stage, shared across the K trees (sklearn-style)
      const idx = this.subsample < 1
        ? subsampleIndices(n, this.subsample, this.random)
        : Array.from({ length: n }, (_, i) => i);

      const stageTrees = [];
      for (let k = 0; k < K; k++) {
        const residuals = y.map((yi, i) => (yi === k ? 1 : 0) - P[i][k]);

        // Friedman's multiclass Newton step:
        // gamma = (K-1)/K * sum(r) / sum(|r| * (1 - |r|))
        const { tree, update } = this._fitStageTree(X, residuals, idx, (samples) => {
          let num = 0;
          let den = 0;
          for (const i of samples) {
            const r = residuals[i];
            num += r;
            den += Math.abs(r) * (1 - Math.abs(r));
          }
          return ((K - 1) / K) * (num / Math.max(den, EPS));
        });
        stageTrees.push(tree);

        for (let i = 0; i < n; i++) {
          F[i][k] += this.learningRate * update[i];
        }
      }
      this.trees.push(stageTrees);

      const deviance = -y.reduce((s, yi, i) => {
        const p = Math.max(softmaxRow(F[i])[yi], EPS);
        return s + Math.log(p);
      }, 0) / n;
      this.lossHistory_.push(deviance);
    }
  }

  _decisionFunction(X) {
    const data = preparePredict(X, this.columns);

    if (this.task === "regression" || this.nClasses_ === 2) {
      return data.map((row) => {
        let f = this.init_;
        for (const tree of this.trees) {
          f += this.learningRate * findLeaf(tree.tree.root, row).value;
        }
        return f;
      });
    }

    // multiclass: one score per class
    return data.map((row) => {
      const f = this.init_.slice();
      for (const stageTrees of this.trees) {
        for (let k = 0; k < f.length; k++) {
          f[k] += this.learningRate * findLeaf(stageTrees[k].tree.root, row).value;
        }
      }
      return f;
    });
  }

  _predictRaw(X) {
    this._ensureFitted("predict");
    return this._decisionFunction(X);
  }

  _predictProba(X) {
    this._ensureFitted("predictProba");
    const scores = this._decisionFunction(X);
    if (this.nClasses_ === 2) {
      return scores.map((f) => {
        const p = sigmoid(f);
        return [1 - p, p];
      });
    }
    return scores.map(softmaxRow);
  }

  _predictClassIndices(X) {
    return this._predictProba(X).map((probs) => {
      let best = 0;
      for (let k = 1; k < probs.length; k++) {
        if (probs[k] > probs[best]) best = k;
      }
      return best;
    });
  }

  _computeFeatureImportances(nFeatures) {
    const importances = new Array(nFeatures).fill(0);
    const allTrees = this.task === "regression" || this.nClasses_ === 2
      ? this.trees
      : this.trees.flat();

    for (const tree of allTrees) {
      const treeImportances = tree.tree.featureImportances;
      for (let f = 0; f < nFeatures; f++) {
        importances[f] += treeImportances[f];
      }
    }

    const total = importances.reduce((a, b) => a + b, 0);
    this._featureImportances = total > 0
      ? importances.map((v) => v / total)
      : importances;
  }

  get featureImportances() {
    this._ensureFitted("featureImportances");
    return this._featureImportances;
  }
}

/**
 * Gradient boosting for regression (squared-error loss)
 *
 * @example
 * const gbr = new GradientBoostingRegressor({ nEstimators: 200, learningRate: 0.05 });
 * gbr.fit(X, y);
 * const predictions = gbr.predict(Xnew);
 */
export class GradientBoostingRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.gb = new GradientBoostingBase({ ...opts, task: "regression" });
  }

  /**
   * Fit the regressor on training data.
   * @param {Array<Array<number>>} X - Feature matrix (n samples × p features)
   * @param {Array<number>} [y] - Target values
   * @returns {this} The fitted estimator (for chaining)
   */
  fit(X, y = null) {
    this.gb.fit(X, y);
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values for samples in X.
   * @param {Array<Array<number>>} X - Feature matrix (n samples × p features)
   * @returns {Array<number>} Predicted target values
   */
  predict(X) {
    return this.gb._predictRaw(X);
  }

  get featureImportances() {
    return this.gb.featureImportances;
  }

  /** Per-stage training loss (MSE) */
  get lossHistory() {
    return this.gb.lossHistory_.slice();
  }
}

/**
 * Gradient boosting for classification (logistic / multinomial deviance)
 *
 * @example
 * const gbc = new GradientBoostingClassifier({ nEstimators: 100 });
 * gbc.fit({ X: ['bill_length', 'bill_depth'], y: 'species', data });
 * const labels = gbc.predict({ data: newData });
 */
export class GradientBoostingClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.gb = new GradientBoostingBase({ ...opts, task: "classification" });
  }

  /**
   * Fit the classifier on training data.
   * @param {Array<Array<number>>} X - Feature matrix (n samples × p features)
   * @param {Array<number>|Array<string>} [y] - Class labels
   * @returns {this} The fitted estimator (for chaining)
   */
  fit(X, y = null) {
    const prepared = prepareDataset(X, y);

    this._extractLabelEncoder(prepared);
    const { numericY, classes } = this._getClasses(prepared.y, false);
    this.classes_ = classes;

    // Map labels to dense indices 0..K-1 in classes_ order (numericY may be
    // raw numeric labels like [1, 2] rather than indices)
    const classIndex = new Map(classes.map((c, i) => [c, i]));
    const yIdx = this.labelEncoder_
      ? numericY
      : numericY.map((v, i) => {
        const label = typeof classes[0] === "string" ? prepared.y[i] : v;
        return classIndex.get(label) !== undefined ? classIndex.get(label) : v;
      });

    this.gb._fitPrepared(prepared.X, yIdx, prepared.columns);
    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   * @param {Array<Array<number>>} X - Feature matrix (n samples × p features)
   * @returns {Array<number>|Array<string>} Predicted class labels
   */
  predict(X) {
    this._ensureFitted("predict");
    const indices = this.gb._predictClassIndices(X);
    return indices.map((i) => this.classes_[i]);
  }

  /**
   * Predict class probabilities
   * @returns {Array<Object>} One object per row keyed by class label
   */
  predictProba(X) {
    this._ensureFitted("predictProba");
    const proba = this.gb._predictProba(X);
    return proba.map((row) => {
      const dist = {};
      this.classes_.forEach((label, k) => {
        dist[label] = row[k];
      });
      return dist;
    });
  }

  get featureImportances() {
    return this.gb.featureImportances;
  }

  /** Per-stage training loss (deviance) */
  get lossHistory() {
    return this.gb.lossHistory_.slice();
  }
}

export default {
  GradientBoostingRegressor,
  GradientBoostingClassifier,
};
