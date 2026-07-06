// ---
// title: Multinomial classification with the GLM
// id: ds-multinomial-classification
// ---

// %% [markdown]
/*
# Multinomial classification with the GLM

`ds.stats.GLM` is a generalized linear model, and with the binomial family it
does logistic regression. Out of the box that is a two-class model, but there
are two standard ways to stretch it to three or more classes:

- **One-vs-rest (OVR)** fits one binary classifier per class ("this class or
  not") and compares their scores.
- **True multinomial** fits all classes jointly with a softmax link.

This notebook runs both on the same small iris-style dataset and shows how
their probabilities differ.
*/

// %% [javascript]

import ds from 'https://esm.sh/@tangent.to/ds';

// The table API takes plain row objects (no DataFrame needed). Three species,
// four measurements each.
const irisRows = [
  { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: 'versicolor' },
  { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: 'versicolor' },
  { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: 'versicolor' },
  { sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 5.5, sepal_width: 2.6, petal_length: 3.5, petal_width: 1.0, species: 'versicolor' },
  { sepal_length: 6.0, sepal_width: 2.2, petal_length: 4.0, petal_width: 1.0, species: 'versicolor' },
  { sepal_length: 6.7, sepal_width: 3.1, petal_length: 4.7, petal_width: 1.5, species: 'virginica' },
  { sepal_length: 5.8, sepal_width: 2.7, petal_length: 4.1, petal_width: 1.0, species: 'virginica' },
  { sepal_length: 6.3, sepal_width: 2.5, petal_length: 5.0, petal_width: 1.9, species: 'virginica' },
];

const features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];

({
  rows: irisRows.length,
  features,
  classes: [...new Set(irisRows.map((r) => r.species))],
});

// %% [markdown]
/*
## Method 1: one-vs-rest

Setting `multiclass: 'ovr'` tells the GLM to fit one binomial model per class.
We pass a table-style object (`{ data, X, y }`) where `y` names the label
column. `predict` returns class labels; passing `{ type: 'proba' }` as the
second argument returns a probability object per row (renormalised to sum to
one across the classes).
*/

// %% [javascript]

const ovrModel = new ds.stats.GLM({ family: 'binomial', multiclass: 'ovr' });
ovrModel.fit({ data: irisRows, X: features, y: 'species' });

const ovrClasses = ovrModel._classes;
const ovrPreds = ovrModel.predict({ data: irisRows }, { type: 'class' });
const ovrProba = ovrModel.predict({ data: irisRows }, { type: 'proba' });

({
  classes: ovrClasses,               // [setosa, versicolor, virginica]
  predictions_first5: ovrPreds.slice(0, 5),
  probability_row0: ovrProba[0],     // setosa ~0.93, versicolor ~0.07
});

// %% [markdown]
/*
## Method 2: true multinomial

`multiclass: 'multinomial'` fits the K classes jointly with a softmax link,
optimising K-1 parameter vectors against a reference class. The call site is
identical; only the strategy changes. Because softmax is normalised by
construction, each row's probabilities sum to exactly one, and they tend to be
sharper (better calibrated) than the OVR scores.
*/

// %% [javascript]

const mnModel = new ds.stats.GLM({ family: 'binomial', multiclass: 'multinomial' });
mnModel.fit({ data: irisRows, X: features, y: 'species' });

const mnPreds = mnModel.predict({ data: irisRows }, { type: 'class' });
const mnProba = mnModel.predict({ data: irisRows }, { type: 'proba' });
const mnRow0 = mnProba[0];

({
  isMultinomial: mnModel._isMultinomial,   // true
  predictions_first5: mnPreds.slice(0, 5),
  probability_row0: mnRow0,                // setosa ~1.0
  row0_sums_to: Object.values(mnRow0).reduce((a, b) => a + b, 0), // 1
});

// %% [markdown]
/*
## Which to reach for

Both recover the labels on this clean data; the difference is in the
probabilities. OVR is simple and scales to many classes or imbalanced data,
but its per-class scores are estimated independently and only sum to one after
renormalisation. True multinomial optimises the classes together, so its
probabilities are inherently normalised and usually better calibrated — at the
cost of a joint fit. For a handful of classes, prefer multinomial; for many
classes or a quick baseline, OVR is a reasonable default.
*/

// %% [javascript]

({
  ovr: {
    strategy: 'one-vs-rest',
    usage: "new ds.stats.GLM({ family: 'binomial', multiclass: 'ovr' })",
    probabilities_normalised: 'after renormalisation',
  },
  multinomial: {
    strategy: 'softmax (joint)',
    usage: "new ds.stats.GLM({ family: 'binomial', multiclass: 'multinomial' })",
    probabilities_normalised: 'by construction',
  },
});
