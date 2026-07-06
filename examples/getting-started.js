// ---
// title: Getting started with tangent/ds
// id: ds-getting-started
// ---

// %% [markdown]
/*
# Getting started

`@tangent.to/ds` is a browser-first data-science toolkit. It bundles
descriptive statistics, hypothesis tests, multivariate analysis, and machine
learning behind an API that reads like scikit-learn and R: estimators are
objects you `fit` and then `transform` or `predict`, and functional helpers
take plain numeric arrays. The whole library runs in the browser with no
build step and no native dependencies.

The default export is namespaced: `ds.core`, `ds.stats`, `ds.mva`, and
`ds.ml`. We will walk one short flow end to end: describe a dataset, test a
difference between groups, reduce dimensions with PCA, cluster, and fit a
linear model.
*/

// %% [javascript]

import ds from 'https://esm.sh/@tangent.to/ds';

const math = ds.core.math;

// A quick sanity check that the namespaces are present.
({
  core: Object.keys(ds.core),
  stats_has_GLM: typeof ds.stats.GLM,
  mva_has_PCA: typeof ds.mva.PCA,
  ml_has_KMeans: typeof ds.ml.KMeans,
});

// %% [markdown]
/*
## A self-contained dataset

We use a small flower dataset in the spirit of Fisher's iris: twelve rows,
two species, three numeric measurements each (in centimetres). Keeping it
inline means the notebook has no file loading and every number below is
reproducible. The two species are chosen to be clearly separable, which
makes the later steps easy to read.
*/

// %% [javascript]

const flowers = [
  { species: 'setosa', petalLength: 1.4, petalWidth: 0.2, sepalLength: 5.1 },
  { species: 'setosa', petalLength: 1.4, petalWidth: 0.2, sepalLength: 4.9 },
  { species: 'setosa', petalLength: 1.3, petalWidth: 0.2, sepalLength: 4.7 },
  { species: 'setosa', petalLength: 1.5, petalWidth: 0.2, sepalLength: 4.6 },
  { species: 'setosa', petalLength: 1.4, petalWidth: 0.2, sepalLength: 5.0 },
  { species: 'setosa', petalLength: 1.7, petalWidth: 0.4, sepalLength: 5.4 },
  { species: 'versicolor', petalLength: 4.7, petalWidth: 1.4, sepalLength: 7.0 },
  { species: 'versicolor', petalLength: 4.5, petalWidth: 1.5, sepalLength: 6.4 },
  { species: 'versicolor', petalLength: 4.9, petalWidth: 1.5, sepalLength: 6.9 },
  { species: 'versicolor', petalLength: 4.0, petalWidth: 1.3, sepalLength: 5.5 },
  { species: 'versicolor', petalLength: 4.6, petalWidth: 1.5, sepalLength: 6.5 },
  { species: 'versicolor', petalLength: 4.5, petalWidth: 1.3, sepalLength: 5.7 },
];

// Pull out the columns we will reuse across the notebook.
const petalLength = flowers.map((f) => f.petalLength);
const petalWidth = flowers.map((f) => f.petalWidth);

({
  rows: flowers.length,
  columns: Object.keys(flowers[0]),
  first_row: flowers[0],
});

// %% [markdown]
/*
## Descriptive statistics

`ds.core.math` provides the numeric primitives: `mean`, `variance` (sample
variance by default), `stddev`, quantiles, and more. `ds.stats` adds
correlation with an inference layer attached. Here petal length and petal
width move together almost perfectly (r about 0.99, p about 5e-11), which is
a useful thing to know before we ask a model to use both.
*/

// %% [javascript]

const plStats = {
  mean: math.mean(petalLength),      // 2.99
  variance: math.variance(petalLength), // 2.64 (sample)
  stddev: math.stddev(petalLength),  // 1.63
};

// pearsonCorrelation returns the coefficient plus a t-test and 95% interval.
const corr = ds.stats.pearsonCorrelation(petalLength, petalWidth);

({
  petalLength: plStats,
  correlation_r: corr.r,
  correlation_p: corr.pValue,
});

// %% [markdown]
/*
## A two-sample t-test

Do the two species differ in petal length? We split the column by species
and run an independent two-sample t-test. The helper returns the t
statistic, the p-value, degrees of freedom, and both group means. These now
match scipy's `ttest_ind` to machine precision. The difference here is
enormous (t about -22.8, p about 6e-10): setosa petals average 1.45 cm
against versicolor's 4.53 cm.
*/

// %% [javascript]

const setosaPetal = flowers
  .filter((f) => f.species === 'setosa')
  .map((f) => f.petalLength);

const versicolorPetal = flowers
  .filter((f) => f.species === 'versicolor')
  .map((f) => f.petalLength);

const tTest = ds.stats.hypothesis.twoSampleTTest(setosaPetal, versicolorPetal);

({
  statistic: tTest.statistic,
  pValue: tTest.pValue,
  df: tTest.df,
  mean_setosa: tTest.mean1,
  mean_versicolor: tTest.mean2,
});

// %% [markdown]
/*
## Principal component analysis

`ds.mva.PCA` is a fit/transform estimator. We fit it on the three numeric
columns as a plain matrix (array of arrays), then read the variance each
component explains. Because the measurements are highly correlated, the
first component alone captures about 97% of the variance, and the first two
together reach about 99.9%. `transform` projects the rows onto the
components, giving low-dimensional scores you could plot.
*/

// %% [javascript]

const matrix = flowers.map((f) => [f.petalLength, f.petalWidth, f.sepalLength]);

const pca = new ds.mva.PCA().fit(matrix);
const pcaSummary = pca.summary();
const scores = pca.transform(matrix);

({
  varianceExplained: pcaSummary.varianceExplained, // [0.967, 0.032, 0.001]
  cumulative: pcaSummary.cumulativeVariance,
  first_two_scores: scores.slice(0, 2),
});

// %% [markdown]
/*
## K-means clustering

`ds.ml.KMeans` follows the same object pattern. With `k: 2` and a fixed seed
for reproducibility, it recovers the two species without ever seeing the
labels: the first six rows land in one cluster and the last six in the
other. `inertia` is the within-cluster sum of squared distances, the
quantity k-means minimises, and a smaller value means tighter clusters.
*/

// %% [javascript]

const kmeans = new ds.ml.KMeans({ k: 2, seed: 42 }).fit(matrix);

({
  labels: kmeans.labels,     // [0,0,0,0,0,0, 1,1,1,1,1,1]
  inertia: kmeans.inertia,   // about 2.94
  iterations: kmeans.iterations,
});

// %% [markdown]
/*
## A linear model with the GLM

Finally a generalized linear model. With the Gaussian family, `ds.stats.GLM`
is ordinary least squares. We regress petal width on petal length and read
the coefficients: an intercept near -0.32 and a slope near 0.38, so each
extra centimetre of petal length predicts about 0.38 cm more petal width.
The `coefficients` array is ordered intercept first, then one slope per
feature.
*/

// %% [javascript]

const glm = new ds.stats.GLM({ family: 'gaussian' })
  .fit(flowers.map((f) => [f.petalLength]), petalWidth);

({
  intercept: glm.coefficients[0],  // -0.32
  slope_petalLength: glm.coefficients[1], // 0.38
});
