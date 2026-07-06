/**
 * Stats module exports
 */

import { normal, uniform, gamma, beta, chisq, qchisq } from './distribution.js';
import {
  oneSampleTTest as oneSampleTTestFn,
  twoSampleTTest as twoSampleTTestFn,
  pairedTTest as pairedTTestFn,
  chiSquareTest as chiSquareTestFn,
  oneWayAnova as oneWayAnovaFn,
  tukeyHSD as tukeyHSDFn,
  mannWhitneyU as mannWhitneyUFn,
  kruskalWallis as kruskalWallisFn,
  cohensD,
  etaSquared,
  omegaSquared,
  bonferroni,
  holmBonferroni,
  fdr,
  leveneTest,
  pearsonCorrelation,
  spearmanCorrelation,
  fisherExactTest
} from './tests.js';

import { GLM } from './estimators/GLM.js';
import {
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  TukeyHSD
} from './estimators/tests.js';

import {
  compareModels,
  likelihoodRatioTest,
  pairwiseLRT,
  modelSelectionPlot,
  aicWeightPlot,
  coefficientComparisonPlot,
  crossValidate,
  crossValidateModels
} from './model_comparison.js';

// Alias classes under camelCase names for ergonomic construction.
// These are 1:1 aliases of the exported estimator classes above; they are
// excluded from generated docs (@ignore) since deno-doc cannot derive an
// explicit type for a bare identifier alias in a .js file. See the PascalCase
// classes (OneSampleTTest, TwoSampleTTest, ...) for the documented API.
/**
 * camelCase alias of the {@link OneSampleTTest} estimator class.
 * @ignore
 */
const oneSampleTTest = OneSampleTTest;
/**
 * camelCase alias of the {@link TwoSampleTTest} estimator class.
 * @ignore
 */
const twoSampleTTest = TwoSampleTTest;
/**
 * camelCase alias of the {@link ChiSquareTest} estimator class.
 * @ignore
 */
const chiSquareTest = ChiSquareTest;
/**
 * camelCase alias of the {@link OneWayAnova} estimator class.
 * @ignore
 */
const oneWayAnova = OneWayAnova;
/**
 * camelCase alias of the {@link TukeyHSD} estimator class.
 * @ignore
 */
const tukeyHSD = TukeyHSD;

// Alias functional tests for camelCase export (1:1 aliases; @ignore as above).
/**
 * Paired-sample t-test (functional alias of the exported `pairedTTest`).
 * @ignore
 */
const pairedTTest = pairedTTestFn;
/**
 * Mann-Whitney U rank-sum test (functional alias).
 * @ignore
 */
const mannWhitneyU = mannWhitneyUFn;
/**
 * Kruskal-Wallis one-way test on ranks (functional alias).
 * @ignore
 */
const kruskalWallis = kruskalWallisFn;

/**
 * Functional hypothesis-test helpers grouped under a single namespace for
 * direct (non-estimator) usage. Each property is the functional form of the
 * corresponding test (`oneSampleTTest`, `twoSampleTTest`, `pairedTTest`,
 * `chiSquareTest`, `oneWayAnova`, `tukeyHSD`, `mannWhitneyU`, `kruskalWallis`).
 */
const hypothesis = {
  oneSampleTTest: oneSampleTTestFn,
  twoSampleTTest: twoSampleTTestFn,
  pairedTTest: pairedTTestFn,
  chiSquareTest: chiSquareTestFn,
  oneWayAnova: oneWayAnovaFn,
  tukeyHSD: tukeyHSDFn,
  mannWhitneyU: mannWhitneyUFn,
  kruskalWallis: kruskalWallisFn
};

export {
  // Distributions
  normal,
  uniform,
  gamma,
  beta,
  chisq,
  qchisq,

  // Hypothesis test helper namespace (functional)
  hypothesis,

  // Generalized Linear Models (GLM and GLMM)
  GLM,

  // Model comparison and selection
  compareModels,
  likelihoodRatioTest,
  pairwiseLRT,
  modelSelectionPlot,
  aicWeightPlot,
  coefficientComparisonPlot,
  crossValidate,
  crossValidateModels,

  // Hypothesis test estimator classes
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  TukeyHSD,
  oneSampleTTest,
  twoSampleTTest,
  chiSquareTest,
  oneWayAnova,
  tukeyHSD,

  // Additional statistical tests (functional only)
  pairedTTest,
  mannWhitneyU,
  kruskalWallis,

  // Effect sizes
  cohensD,
  etaSquared,
  omegaSquared,

  // Multiple testing corrections
  bonferroni,
  holmBonferroni,
  fdr,

  // Assumption testing
  leveneTest,

  // Correlation tests
  pearsonCorrelation,
  spearmanCorrelation,

  // Contingency tables
  fisherExactTest
};
