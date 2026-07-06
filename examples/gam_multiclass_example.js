// ---
// title: Multi-class GAM classification
// id: ds-gam-multiclass
// ---

// %% [markdown]
/*
A generalized additive model (GAM) replaces the straight-line terms of a
linear model with smooth, wiggly functions of each predictor, so it can
capture curved relationships while staying interpretable. `ds.ml.GAMClassifier`
wraps that idea for classification: it fits penalised regression splines and,
for more than two classes, a multinomial (softmax) link.

This notebook covers three things: how to control the smoothing-parameter
search, how to fit and read a three-class model, and that binary
classification still works through the same object.
*/

// %% [javascript]

import ds from 'https://esm.sh/@tangent.to/ds';

// A quick check that the estimator we need is present.
({
  has_GAMClassifier: typeof ds.ml.GAMClassifier,
  has_GAMRegressor: typeof ds.ml.GAMRegressor,
});

// %% [markdown]
/*
## Controlling the smoothing search

With `smoothMethod: 'GCV'` the model chooses each spline's smoothing penalty
by minimising generalized cross-validation over a grid of candidate values.
You can steer that grid: `lambdaMin` and `lambdaMax` bound the penalty and
`nSteps` sets how many points are tried in log-space (defaults are `1e-8`,
`1e4`, and `20`). A wider or finer grid trades compute for a better fit.
*/

// %% [javascript]

const gcvGam = new ds.ml.GAMClassifier({
  nSplines: 10,
  basis: 'cr',        // cubic regression spline basis
  smoothMethod: 'GCV',
  lambdaMin: 1e-6,    // custom lower bound
  lambdaMax: 1e3,     // custom upper bound
  nSteps: 30,         // finer grid than the default 20
});

// The estimator stores its resolved configuration on `.params`.
({
  smoothMethod: gcvGam.params.smoothMethod,
  lambdaMin: gcvGam.params.lambdaMin,
  lambdaMax: gcvGam.params.lambdaMax,
  nSteps: gcvGam.params.nSteps,
});

// %% [markdown]
/*
## A three-class dataset

We use a small iris-style dataset: thirty rows across three species, each
with two continuous measurements. The classes are laid out from small to
large feature values so the smooths have a clear signal to pick up. `X` is a
plain matrix (array of arrays) and `y` is an array of string labels.
*/

// %% [javascript]

const Xtrain = [
  // setosa - small values
  [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
  [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
  // versicolor - medium values
  [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
  [5.7, 2.8], [6.3, 3.3], [4.9, 2.4], [6.6, 2.9], [5.2, 2.7],
  // virginica - large values
  [6.3, 3.3], [5.8, 2.7], [7.1, 3.0], [6.3, 2.9], [6.5, 3.0],
  [7.6, 3.0], [4.9, 2.5], [7.3, 2.9], [6.7, 2.5], [7.2, 3.6],
];

const ytrain = [
  'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
  'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
  'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
  'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
  'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
  'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
];

({
  samples: Xtrain.length,
  features: Xtrain[0].length,
  classes: [...new Set(ytrain)],
});

// %% [markdown]
/*
## Fitting the multi-class model

Here we fix the smoothing penalty with `lambda: 0.1` instead of searching for
it, which keeps the fit fast and deterministic. After `fit`, the underlying
model exposes `K` (the number of classes), the sorted `classes`, and `coef`:
a multinomial GAM fits K-1 coefficient vectors, one class acting as the
reference.
*/

// %% [javascript]

const gam3 = new ds.ml.GAMClassifier({ nSplines: 5, basis: 'cr', lambda: 0.1 });
gam3.fit(Xtrain, ytrain);

({
  nClasses: gam3.gam.K,           // 3
  classNames: gam3.gam.classes,   // [setosa, versicolor, virginica]
  coefVectors: gam3.gam.coef.length, // 2 = K - 1
});

// %% [markdown]
/*
## Reading the summary

`summary()` reports the family and link (multinomial / softmax for three
classes), the overall training accuracy, and a per-class accuracy breakdown.
On this tiny, overlapping dataset the model gets about 77% of the training
rows right, and does best on the well-separated setosa class.
*/

// %% [javascript]

const gam3Summary = gam3.summary();

({
  call: gam3Summary.call,
  family: gam3Summary.family,             // multinomial
  link: gam3Summary.link,                 // softmax
  trainingAccuracy: gam3Summary.trainingAccuracy,   // ~0.767
  perClassAccuracy: gam3Summary.perClassAccuracy,   // setosa 0.9, others 0.7
});

// %% [markdown]
/*
The per-class accuracy is easier to compare as bars: the well-separated setosa
class is learned almost perfectly, the overlapping classes less so.
*/

// %% [javascript]

const perClassBars = Object.entries(gam3Summary.perClassAccuracy).map(
  ([cls, acc]) => ({ class: cls, accuracy: acc }),
);
const plot_gam3Acc = Plot.plot({
  x: { label: 'Class' },
  y: { label: 'Training accuracy', domain: [0, 1], grid: true },
  color: { legend: true },
  marks: [
    Plot.barY(perClassBars, { x: 'class', y: 'accuracy', fill: 'class' }),
    Plot.ruleY([0]),
  ],
});
plot_gam3Acc;

// %% [markdown]
/*
## Predictions and probabilities

`predict` returns the most likely class label per row; `predictProba` returns
a probability object keyed by class name. We score three held-out points, one
aimed at each class. The setosa point is called confidently; the two larger
points are genuinely ambiguous between versicolor and virginica, and the model
leans virginica for both — a fair reflection of how much these two classes
overlap in just two measurements.
*/

// %% [javascript]

const Xtest = [
  [5.0, 3.5], // aimed at setosa
  [6.5, 3.0], // aimed at versicolor
  [7.0, 3.0], // aimed at virginica
];

const preds3 = gam3.predict(Xtest);
const proba3 = gam3.predictProba(Xtest);

({
  predictions: preds3,
  probabilities: proba3,
});

// %% [markdown]
/*
Plotting the probability each test point gets for every class makes the
model's confidence legible: setosa is called decisively, while the two larger
points spread their mass between versicolor and virginica.
*/

// %% [javascript]

const probaLong = proba3.flatMap((row, i) =>
  Object.entries(row).map(([cls, p]) => ({
    point: `test ${i} (predicted ${preds3[i]})`,
    class: cls,
    prob: p,
  })),
);
const plot_gam3Proba = Plot.plot({
  marginLeft: 90,
  x: { label: 'Probability', domain: [0, 1] },
  y: { label: 'Class' },
  fy: { label: null },
  color: { legend: true },
  marks: [
    Plot.barX(probaLong, { x: 'prob', y: 'class', fy: 'point', fill: 'class' }),
    Plot.ruleX([0]),
  ],
});
plot_gam3Proba;

// %% [markdown]
/*
## Binary classification still works

The same class handles two classes without any special flags: with only two
labels it fits a single coefficient vector (K-1 = 1) and a logistic link. Here
two well-separated clusters are classified with near-certain probabilities.
*/

// %% [javascript]

const Xbin = [
  [1, 2], [1.5, 1.8], [1.2, 2.1], // class 0
  [5, 6], [5.5, 5.8], [5.2, 6.1], // class 1
];
const ybin = [0, 0, 0, 1, 1, 1];

const gamBin = new ds.ml.GAMClassifier({ nSplines: 3 });
gamBin.fit(Xbin, ybin);

const probaBin = gamBin.predictProba([[1, 2], [5, 6]]);

({
  nClasses: gamBin.gam.K,          // 2
  coefVectors: gamBin.gam.coef.length, // 1
  proba_low: probaBin[0],          // ~{0: 1, 1: 0}
  proba_high: probaBin[1],         // ~{0: 0, 1: 1}
});

// %% [markdown]
/*
The two well-separated clusters are classified with near-certain probabilities
— each point puts almost all of its mass on the correct class.
*/

// %% [javascript]

const binLong = [probaBin[0], probaBin[1]].flatMap((row, i) =>
  Object.entries(row).map(([cls, p]) => ({
    point: i === 0 ? 'low point [1, 2]' : 'high point [5, 6]',
    class: `class ${cls}`,
    prob: p,
  })),
);
const plot_gamBin = Plot.plot({
  marginLeft: 110,
  x: { label: 'Probability', domain: [0, 1] },
  y: { label: 'Predicted class' },
  fy: { label: null },
  color: { legend: true },
  marks: [
    Plot.barX(binLong, { x: 'prob', y: 'class', fy: 'point', fill: 'class' }),
    Plot.ruleX([0]),
  ],
});
plot_gamBin;
