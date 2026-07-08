/**
 * Redundancy Analysis (RDA)
 * Constrained ordination - PCA on fitted values from multiple regression
 */

import { Matrix, solveLeastSquares, svd, toMatrix as _toMatrix } from '../core/linalg.js';
import * as pca from './pca.js';
import { mean } from '../core/math.js';
import { prepareX, attachSourceRows } from '../core/table.js';
import {
  eigenvaluePowers,
  normalizeScaling,
  scaleConstraintScores,
  scaleOrdination,
  toLoadingObjects,
  toScoreObjects,
} from './scaling.js';

/**
 * Fit RDA model.
 *
 * @param {Array<Array<number>>} Y - Response matrix (n x q)
 * @param {Array<Array<number>>} X - Explanatory matrix (n x p)
 * @param {Object} options
 * @param {boolean} [options.scale=false] - Standardise response variables before regression.
 * @param {boolean} [options.constrained=true] - When true, perform PCA on fitted values (constrained ordination); when false, perform PCA on residuals (unconstrained ordination).
 * @returns {Object} RDA model
 */
export function fit(Y, X, options = {}) {
  let scale = options.scale ?? false;
  let scaling = options.scaling ?? 2;
  let constrained = options.constrained ?? true;
  // Normalize naOmit parameter (naOmit is primary, omit_missing is alias)
  let naOmit = options.omit_missing !== undefined
    ? options.omit_missing
    : (options.naOmit !== undefined ? options.naOmit : true);
  let responseMatrix = Y;
  let predictorMatrix = X;
  let sourcePrepared = null;
  let responseNames = Array.isArray(options.responseNames)
    ? options.responseNames.map((name) => String(name))
    : null;
  let predictorNames = Array.isArray(options.predictorNames)
    ? options.predictorNames.map((name) => String(name))
    : null;

  if (
    Y &&
    typeof Y === 'object' &&
    !Array.isArray(Y) &&
    (Y.data || Y.response || Y.responses || Y.predictors || Y.Y)
  ) {
    const opts = Y;
    const data = opts.data;
    const responseCols = opts.response || opts.responses || opts.Y;
    const predictorCols = opts.predictors || opts.X;
    if (!data || !responseCols || !predictorCols) {
      throw new Error('RDA.fit requires data, response columns, and predictor columns.');
    }
    // Handle both naOmit (primary) and omit_missing (alias)
    naOmit = opts.omit_missing !== undefined
      ? opts.omit_missing
      : (opts.naOmit !== undefined ? opts.naOmit : naOmit);
    scale = opts.scale !== undefined ? opts.scale : scale;
    scaling = opts.scaling !== undefined ? opts.scaling : scaling;
    constrained = opts.constrained !== undefined ? opts.constrained : constrained;

    const responseList = Array.isArray(responseCols) ? responseCols : [responseCols];
    const predictorList = Array.isArray(predictorCols) ? predictorCols : [predictorCols];

    const responsePrepInitial = prepareX({
      columns: responseList,
      data,
      naOmit: naOmit,
    });
    const predictorPrep = prepareX({
      columns: predictorList,
      data: responsePrepInitial.rows,
      naOmit: naOmit,
    });
    const responsePrepAligned = prepareX({
      columns: responseList,
      data: predictorPrep.rows,
      naOmit: false,
    });

    responseMatrix = responsePrepAligned.X;
    predictorMatrix = predictorPrep.X;
    sourcePrepared = predictorPrep;
    responseNames = responsePrepAligned.columns.map((name) => String(name));
    predictorNames = predictorPrep.columns.map((name) => String(name));
  }

  if (!responseMatrix || !predictorMatrix) {
    throw new Error('RDA.fit requires response and predictor matrices.');
  }

  const appliedScaling = normalizeScaling(scaling);

  const responseData = responseMatrix.map((row) => Array.isArray(row) ? row : [row]);
  const explData = predictorMatrix.map((row) => Array.isArray(row) ? row : [row]);

  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;

  if (n !== explData.length) {
    throw new Error('Y and X must have same number of rows');
  }

  if (n < p + 2) {
    throw new Error('Need more samples than explanatory variables');
  }

  const YMeans = [];
  const XMeans = [];

  for (let j = 0; j < q; j++) {
    const col = responseData.map((row) => row[j]);
    YMeans.push(mean(col));
  }

  for (let j = 0; j < p; j++) {
    const col = explData.map((row) => row[j]);
    XMeans.push(mean(col));
  }

  let YCentered = responseData.map((row) => row.map((val, j) => val - YMeans[j]));

  const XCentered = explData.map((row) => row.map((val, j) => val - XMeans[j]));

  // Apply scaling to Y if requested
  let YSds = null;
  if (scale) {
    YSds = [];
    for (let j = 0; j < q; j++) {
      const col = YCentered.map((row) => row[j]);
      const sd = col.reduce((sum, val) => sum + val * val, 0) / n;
      YSds.push(Math.sqrt(sd));
    }
    YCentered = YCentered.map((row) => row.map((val, j) => YSds[j] > 0 ? val / YSds[j] : 0));
  }

  const YFitted = [];
  const YResiduals = [];
  const coefficients = [];

  for (let j = 0; j < q; j++) {
    const yCol = YCentered.map((row) => row[j]);

    const XMat = new Matrix(XCentered);
    const yVec = Matrix.columnVector(yCol);
    const betaVec = solveLeastSquares(XMat, yVec);
    const beta = betaVec.to1DArray();

    coefficients.push(beta);

    const fitted = [];
    const residuals = [];
    for (let i = 0; i < n; i++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * beta[k];
      }
      fitted.push(yhat);
      residuals.push(yCol[i] - yhat);
    }

    YFitted.push(fitted);
    YResiduals.push(residuals);
  }

  const fittedMatrix = [];
  const residualMatrix = [];
  for (let i = 0; i < n; i++) {
    const fittedRow = [];
    const residualRow = [];
    for (let j = 0; j < q; j++) {
      fittedRow.push(YFitted[j][i]);
      residualRow.push(YResiduals[j][i]);
    }
    fittedMatrix.push(fittedRow);
    residualMatrix.push(residualRow);
  }

  const targetMatrix = constrained ? fittedMatrix : residualMatrix;

  const pcaModel = pca.fit(targetMatrix, {
    scale: false,
    center: false,
    scaling: appliedScaling,
    columns: responseNames || undefined,
  });

  const rawSiteMatrix = pcaModel.rawScores;
  const rawLoadingsMatrix = pcaModel.rawLoadings;

  const ordination = scaleOrdination({
    rawSites: rawSiteMatrix,
    rawLoadings: rawLoadingsMatrix,
    eigenvalues: pcaModel.eigenvalues,
    singularValues: pcaModel.singularValues,
    scaling: appliedScaling,
  });

  const responseNamesFinal = responseNames && responseNames.length === rawLoadingsMatrix.length
    ? responseNames
    : Array.from(
      { length: rawLoadingsMatrix.length },
      (_, idx) => responseNames?.[idx] ?? `Resp${idx + 1}`,
    );

  const predictorNamesFinal = predictorNames && predictorNames.length === p
    ? predictorNames
    : Array.from({ length: p }, (_, idx) => predictorNames?.[idx] ?? `Pred${idx + 1}`);

  const scoresObjects = toScoreObjects(ordination.scores, 'rda');
  const loadingsObjects = toLoadingObjects(ordination.loadings, responseNamesFinal, 'rda');

  let rawConstraintMatrix = [];
  let scaledConstraintMatrix = [];
  let constraintObjects = [];
  if (constrained) {
    rawConstraintMatrix = solveLeastSquares(XCentered, rawSiteMatrix).to2DArray();
    scaledConstraintMatrix = scaleConstraintScores(rawConstraintMatrix, {
      loadingFactors: ordination.loadingFactors,
      eigenvalues: pcaModel.eigenvalues,
      scaling: appliedScaling,
    });
    constraintObjects = toLoadingObjects(scaledConstraintMatrix, predictorNamesFinal, 'rda');
  }

  let totalInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      totalInertia += YCentered[i][j] ** 2;
    }
  }
  totalInertia /= n;

  let explainedInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      explainedInertia += fittedMatrix[i][j] ** 2;
    }
  }
  explainedInertia /= n;

  const constrainedVariance = explainedInertia / totalInertia;
  const residualInertia = totalInertia - explainedInertia;

  // Degrees of freedom for the global permutation test (vegan anova.cca
  // convention): the model df is the numerical rank of the centred constraints
  // (= number of canonical constraints; equals p when X is full column rank),
  // and the residual df is n - rank - 1 (the extra 1 for the centring of Y).
  const constraintRank = constrained ? numericRank(XCentered) : 0;
  const dfModel = constraintRank;
  const dfResidual = n - dfModel - 1;
  const pseudoF = (constrained && dfModel > 0 && dfResidual > 0 && residualInertia > 0)
    ? (explainedInertia / dfModel) / (residualInertia / dfResidual)
    : NaN;

  const predictorCorrelations = constraintObjects;

  const model = {
    scores: scoresObjects,
    loadings: loadingsObjects,
    constraintScores: constraintObjects,
    eigenvalues: pcaModel.eigenvalues,
    varianceExplained: pcaModel.varianceExplained,
    constrainedVariance,
    coefficients,
    YMeans,
    XMeans,
    n,
    p,
    q,
    rawScores: rawSiteMatrix,
    rawLoadings: rawLoadingsMatrix,
    rawFitted: fittedMatrix,
    rawResiduals: residualMatrix,
    rawConstraintScores: rawConstraintMatrix,
    siteFactors: ordination.siteFactors,
    loadingFactors: ordination.loadingFactors,
    scaling: appliedScaling,
    exponent: ordination.exponent,
    singularValues: pcaModel.singularValues,
    components: pcaModel.components,
    constrained: !!constrained,
    // Inertia decomposition + degrees of freedom for permutationTest() below.
    totalInertia,
    explainedInertia,
    residualInertia,
    dfModel,
    dfResidual,
    pseudoF,
  };

  model.responseNames = responseNamesFinal;
  model.predictorNames = predictorNamesFinal;

  model.canonicalScores = scoresObjects;
  model.canonicalLoadings = loadingsObjects;
  model.predictorCorrelations = predictorCorrelations;

  // Retain the centred (and, if requested, scaled) response matrix and the
  // centred constraints so permutationTest() can refit permuted responses
  // without re-preparing the data. Non-enumerable so JSON/persistence skip it.
  Object.defineProperty(model, '_work', {
    value: { Y: YCentered, X: XCentered },
    enumerable: false,
  });

  // Reference to the filtered source rows (+ original indices) for plot
  // helpers (colorBy by column name, and realigning external per-row arrays);
  // non-enumerable so persistence/JSON skips it
  attachSourceRows(model, sourcePrepared);

  return model;
}

/**
 * Numerical rank of a matrix (count of singular values above a scaled tolerance).
 * @param {Array<Array<number>>} M
 * @returns {number}
 */
function numericRank(M) {
  const { s } = svd(M);
  const maxSv = s.reduce((a, b) => Math.max(a, b), 0);
  if (maxSv === 0) return 0;
  const tol = Math.max(M.length, M[0].length) * 2.220446049250313e-16 * maxSv;
  return s.filter((v) => v > tol).length;
}

/**
 * Deterministic PRNG (mulberry32) for reproducible permutations.
 * @param {number} a - 32-bit seed
 * @returns {() => number} uniform [0,1) generator
 */
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Permutation test of the global RDA (equivalent to vegan's
 * `anova.cca(model)`): tests H0 that the constraints explain no more response
 * variance than expected by chance. Under H0 the response rows are exchangeable,
 * so we permute the rows of the (centred) response matrix, refit, and compare the
 * permuted pseudo-F to the observed one. Because a row permutation leaves each
 * response column's total sum of squares unchanged, the total inertia and the
 * degrees of freedom are invariant, so only the constrained inertia is recomputed.
 *
 * The pseudo-F is `(constrained inertia / dfModel) / (residual inertia / dfResidual)`
 * with `dfModel` the rank of the constraints and `dfResidual = n - dfModel - 1`,
 * matching vegan. The p-value uses the standard `(1 + #{F* >= F}) / (nperm + 1)`
 * correction. F, the proportion constrained and the df are divisor-invariant, so
 * they match vegan regardless of its n-1 inertia convention; the reported inertia
 * values use the n-1 divisor to match vegan's "Variance" column directly.
 *
 * @param {Object} model - A constrained RDA model from fit().
 * @param {Object} [options]
 * @param {number} [options.permutations=999] - Number of row permutations.
 * @param {number} [options.seed=42] - Seed for reproducibility.
 * @returns {Object} { pseudoF, pValue, permutations, dfModel, dfResidual,
 *   constrainedInertia, residualInertia, totalInertia, constrainedProportion, eigenvalues }
 */
export function permutationTest(model, options = {}) {
  const permutations = options.permutations ?? 999;
  const seed = (options.seed ?? 42) >>> 0;
  const work = model && model._work;
  if (!work) {
    throw new Error('RDA.permutationTest requires a model from rda.fit() (retained response/constraint matrices are missing).');
  }
  if (!model.constrained) {
    throw new Error('RDA.permutationTest applies to a constrained ordination (constrained: true).');
  }
  const Y = work.Y;
  const X = work.X;
  const n = Y.length;
  const q = Y[0].length;
  const dfModel = model.dfModel;
  const dfResidual = model.dfResidual;
  if (!(dfModel > 0) || !(dfResidual > 0)) {
    throw new Error('RDA.permutationTest: degenerate degrees of freedom (need n > rank(X) + 1).');
  }

  const Xmat = new Matrix(X);
  const XtX = Xmat.transpose().mmul(Xmat);
  // B = (X'X)^-1 X'  (p x n): the projection onto the column space of X.
  const B = solveLeastSquares(Xmat, Matrix.eye(n));

  // Constrained sum of squares = trace(beta' X'X beta) with beta = B * R.
  const explainedSS = (R) => {
    const beta = B.mmul(R);
    const T = beta.transpose().mmul(XtX).mmul(beta);
    let tr = 0;
    for (let j = 0; j < q; j++) tr += T.get(j, j);
    return tr;
  };

  // Total SS of the centred response, invariant under row permutation.
  let totalSS = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < q; j++) totalSS += Y[i][j] * Y[i][j];
  }

  const fStat = (explSS) => (explSS / dfModel) / ((totalSS - explSS) / dfResidual);
  const obsSS = explainedSS(new Matrix(Y));
  const Fobs = fStat(obsSS);

  const rng = mulberry32(seed);
  const idx = Array.from({ length: n }, (_, i) => i);
  const EPS = 1e-9;
  let ge = 0;
  for (let b = 0; b < permutations; b++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    const Rp = new Matrix(idx.map((k) => Y[k]));
    if (fStat(explainedSS(Rp)) >= Fobs - EPS) ge++;
  }
  const pValue = (ge + 1) / (permutations + 1);

  const denom = n - 1;
  return {
    pseudoF: Fobs,
    pValue,
    permutations,
    dfModel,
    dfResidual,
    constrainedInertia: obsSS / denom,
    residualInertia: (totalSS - obsSS) / denom,
    totalInertia: totalSS / denom,
    constrainedProportion: obsSS / totalSS,
    eigenvalues: model.eigenvalues,
  };
}

/**
 * Transform new data using fitted RDA model
 * @param {Object} model - Fitted RDA model
 * @param {Array<Array<number>>} Y - New response data
 * @param {Array<Array<number>>} X - New explanatory data
 * @returns {Array<Object>} Canonical scores
 */
export function transform(model, Y, X) {
  const {
    coefficients,
    YMeans,
    XMeans,
    components,
    singularValues,
    siteFactors,
    scaling,
    eigenvalues,
  } = model;

  const responseData = Y.map((row) => Array.isArray(row) ? row : [row]);
  const explData = X.map((row) => Array.isArray(row) ? row : [row]);

  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;

  // Center data
  const _YCentered = responseData.map((row) => row.map((val, j) => val - YMeans[j]));

  const XCentered = explData.map((row) => row.map((val, j) => val - XMeans[j]));

  // Compute fitted values
  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * coefficients[j][k];
      }
      row.push(yhat);
    }
    fittedMatrix.push(row);
  }

  // Extract loading matrix
  const nAxes = components.length;

  const baseScores = [];
  for (const row of fittedMatrix) {
    const entry = [];
    for (let j = 0; j < nAxes; j++) {
      let sum = 0;
      for (let k = 0; k < q; k++) {
        sum += row[k] * components[j][k];
      }
      entry.push(sum);
    }
    baseScores.push(entry);
  }

  const rawScores = baseScores.map((row) =>
    row.map((val, idx) => {
      const sv = singularValues[idx] ?? 0;
      return sv === 0 ? 0 : val / sv;
    })
  );

  const exponent = scaling === 1 ? 0.5 : 0;
  const siteScaling = siteFactors && siteFactors.length
    ? siteFactors
    : eigenvaluePowers(eigenvalues, exponent);
  const scaledScores = rawScores.map((row) => row.map((val, idx) => val * (siteScaling[idx] ?? 1)));

  return toScoreObjects(scaledScores, 'rda');
}
