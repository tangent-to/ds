// ---
// title: Machine Learning
// id: ds-ml
// ---

// %% [markdown]
/*
# Machine Learning

Until now, we worked as detectives to understand what happened through our experimentations with a statistical inference *detective* mode, with the hat of someone trying to fugure out what happened suring the experiment. Now let's take the role of the *oracle* to predict future occurrences with machine learning. Machine learning models are usually declined in two major categories: classification to predict categories, and regression to poredict continuous numbers. `tangent/ds` currently support KNN, decision trees, random forests and multi-layered perceptrons (sequential neural networks).

The ML workflow is often done in 7 steps.

1. Split data into train/test sets
2. Preprocess features (scaling, encoding)
3. Cross-validate to estimate performance
4. Tune hyperparameters to find the best model settings
5. Fit final model on training data
6. Evaluate on test data (data the model has never seen!)
7. Predict on new observations
*/

// %% [javascript]

// import packages
import ds from '../../dist/index.js';
import * as Plot from 'https://esm.sh/@observablehq/plot';

// data
const penguinsResponse = await fetch(
  'https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json',
);
const penguinsDataRaw = await penguinsResponse.json();
const penguinsData = penguinsDataRaw // there is a row with a "." instead of null in the Sex field
  .map(row => row.Sex === '.' ? { ...row, Sex: null } : row)
  .filter(row => row.Sex);

console.table(penguinsData.slice(0, 5));

// %% [markdown]
/*
## Classification

We have not finished with our lovely penguins. Here we use their numerical characteristics to predict the species, which is a classification task: given measurements (beak length, flipper length, etc.), predict which species a penguin belongs to. This is supervised learning — we know the true labels and train the model to recognize patterns that distinguish species.
*/

// %% [javascript]

const penguinsFeatureFields = [
  "Beak Length (mm)",
  "Beak Depth (mm)",
  "Flipper Length (mm)",
  "Body Mass (g)"
]

// %% [markdown]
/*
### 1. Split and 2. scale

The ML workflow includes splitting data in a training set and a testing set, so that we are able to detect overfitting - if the model performs well on training but poorly on testing, it memorized rather than learned. Scale numeric features is often done using z-scores (standardization), which returns variables with a mean of zero and a variance of 1.

$$
z = (x - mean) / std
$$

Critical for distance-based models (KNN, MLP) so that large-scale features (e.g., body mass in grams) don't dominate small-scale features (e.g., beak length in mm). Importantly, fit the scaler on training data only, then transform both train and test with the same parameters. You will often see tutorials scaling the data on the whole set, but it's a form of test data leakage which must be avoided.
*/

// %% [javascript]

// 1. Split once so you get both arrays and table views
const penguinsSplit = ds.ml.validation.trainTestSplit(
  { data: penguinsData, X: penguinsFeatureFields, y: 'Species' },
  { ratio: 0.7, shuffle: true, seed: 42 }
);

// 2. Fit the scaler on TRAIN rows only, then transform train/test
const penguinScaler = new ds.ml.preprocessing.StandardScaler()
  .fit({ data: penguinsSplit.train.data, columns: penguinsFeatureFields });

const penguinsTrainScaled = penguinScaler.transform({
  data: penguinsSplit.train.data,
  columns: penguinsFeatureFields,
  encoders: penguinsSplit.train.metadata.encoders
});

const penguinsTestScaled = penguinScaler.transform({
  data: penguinsSplit.test.data,
  columns: penguinsFeatureFields,
  encoders: penguinsSplit.train.metadata.encoders
});

// %% [markdown]
/*
### 3. Cross-validation

Cross-validation estimates how well the model will perform on unseen data. It doesn't create a final model, just evaluate what we should expect from our model.

How k-fold CV works with k=5

1. Split data into 5 equal folds
2. Train on 4 folds, test on 1 fold → get accuracy score
3. Rotate which fold is the test set
4. Repeat 5 times → 5 accuracy scores
5. Average the scores for overall performance estimate

Each score is accuracy (fraction of correct predictions) on a different fold. Consistency across folds means a stable model.
*/

// %% [javascript]

const penguinsKnnCV = ds.ml.validation.crossValidate(
    (Xtr, ytr) => new ds.ml.KNNClassifier({ k: 5 }).fit(Xtr, ytr),
    (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)),
    { data: penguinsData, X: penguinsFeatureFields, y: 'Species' },
    { k: 5, shuffle: true }
);
penguinsKnnCV.scores

// %% [markdown]
/*
### 4: Hyperparameter tuning

Hyperparameters are settings we choose before training (not learned from data). This step is optional, and is often discarded for large models.

For KNN

- `k`: Number of neighbors to consider (5, 10, 20, 30..., small numbers being more sensitive to local patterns, risking overfitting)
- `weight`: How to weight neighbors
  - `uniform`: All neighbors vote equally
  - `distance`: Closer neighbors have more influence

Grid search tries all combinations:

- 4 values of k × 2 weight options = 8 total combinations
- For each combination, run 5-fold cross-validation
- Pick the combination with the highest average CV score
*/

// %% [javascript]

// param grid to try
const penguinsParamGrid = {
  k: [5, 10, 20, 30],
  weight: ['uniform', 'distance']
};

// grid-search on the TRAIN set only
const penguinsGrid = ds.ml.tuning.GridSearchCV(
  (Xtr, ytr, params) => new ds.ml.KNNClassifier(params).fit(Xtr, ytr), // the estimator function
  (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)), // the scoring function
  { data: penguinsTrainScaled.data, X: penguinsFeatureFields, y: 'Species' }, // declarative train descriptor
  null,                                                // no separate y array
  penguinsParamGrid,
  { k: 5, shuffle: true, verbose: false }              // CV options
);

console.log('Best params', penguinsGrid.bestParams);
console.log('Best CV accuracy', penguinsGrid.bestScore);

// %% [markdown]
/*
And fit the model with the best hyperparameters.
*/

// %% [javascript]

const penguinsTunedModel = new ds.ml.KNNClassifier(penguinsGrid.bestParams).fit({
    data: penguinsTrainScaled.data,
    X: penguinsFeatureFields,
    y: 'Species',
    encoders: penguinsTrainScaled.metadata.encoders
});

// %% [markdown]
/*
### 5. Predict

The model can predict in different formats:
- **`.predict()`**: Returns class labels (integers or strings if encoder provided)
- **`.predictProba()`**: Returns probabilities for each class (useful for uncertainty quantification)

KNN makes predictions by

1. Finding the k nearest neighbors in the training data (using Euclidean distance)
2. Looking at their labels
3. Voting: majority class wins (or weighted by distance)

For a new penguin with certain measurements, KNN finds its 10 nearest neighbors in the variable space. If 7 are Gentoo, 2 are Adelie and 1 is Chinstrap, it predicts Gentoo with a 70% probability.
*/

// %% [javascript]

const penguinsPredSpecies = penguinsTunedModel.predict({
  data: penguinsTestScaled.data,
  X: penguinsFeatureFields,
  encoders: penguinsTestScaled.metadata.encoders
});
penguinsPredSpecies;

// %% [markdown]
/*
### 6. Evaluate the model

For classifications, the confusion matrix shows where the model succeeds and fails.

- Rows are the true labels (what the penguin actually is).
- Columns are Predicted labels (what the model guessed).
- The diagonal are the correct predictions (darker = more correct).
- The off-diagonal show miscategorization (where the model confused one species for another).

(Sum of diagonal) / (Total predictions) = (48+17+37) / 103 ≈ 99%
*/

// %% [javascript]

const penguinsTrueSpecies = penguinsSplit.test.data.map(row => row.Species);
ds.plot.plotConfusionMatrix(penguinsTrueSpecies, penguinsPredSpecies).show(Plot)

// %% [markdown]
/*
## Regression

Let's leave the penguins behind to demonstrate regression for predicting diamond price, a continuous numer, based on physical characteristics (carat, cut, color, clarity, dimensions).
*/

// %% [javascript]

// Adapted for the browser: arquero is loaded from a CDN instead of a dynamic import.
import * as aq from 'https://esm.sh/arquero';
const url = "https://raw.githubusercontent.com/tidyverse/ggplot2/e594b49fdd5e4d95bf1031edaf6c7ccfc0cdedb0/data-raw/diamonds.csv";
const diamondsTable = await aq.loadCSV(url);
console.table(diamondsTable.objects().slice(0, 6))

// %% [markdown]
/*
Diamonds data has mixed types, which should be carefully treated.

1. Ordinal encoding for `cut` (ordered category):
- Fair < Good < Very Good < Premium < Ideal
- Map to numbers: 0, 1, 2, 3, 4
- Preserves the ordering (better cuts = higher numbers)

2. Fix data type for `depth` stored as string
- Convert "61.5" to the float 61.5

3. One-hot encoding for `color` and `clarity` (nominal categories):
- No natural ordering (D is not "more" than E)
- Create binary columns: color_D, color_E, color_F, etc.
- Example: color="E" → [0, 1, 0, 0, 0, 0, 0]

First, let's fix `cut` and `depth`.
*/

// %% [javascript]

const cutOrder = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'];
const cutToScore = new Map(cutOrder.map((name, idx) => [name, idx]));
const diamondsWithCutScore = diamondsTable
  .objects()
  .filter((row) => cutToScore.has(row.cut))
  .map((row) => ({
    ...row,
    cut: cutToScore.get(row.cut)
  }));

const diamondsDepthNum = diamondsWithCutScore.map(item => ({
  ...item,
  depth: parseFloat(item.depth) // string to float
}));

console.table(diamondsDepthNum.slice(0, 6))

// %% [markdown]
/*
For the rest of the preprocessing, the **recipe API** chains these transformations in a clean pipeline.
*/

// %% [javascript]

const recipeNumeric = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut'];
const recipeCategorical = ['color', 'clarity'];

const diamondsRecipe = ds.ml.recipe({
    data: diamondsDepthNum,
    X: [...recipeNumeric, ...recipeCategorical],
    y: 'price',
  })
  .parseNumeric([...recipeNumeric, 'price'])
  .oneHot(recipeCategorical, { dropFirst: false })
  .scale(recipeNumeric)
  .split({ ratio: 0.8, shuffle: true, seed: 123 });

const diamondsRecipePrepped = diamondsRecipe.prep();

// %% [markdown]
/*
We will run a small MLP neural network with two layers of 32 neurons, a total of 64. This is tiny compared to hundreds of billion parameters of modern LLMs, but let's see if it's enough.
*/

// %% [javascript]

const diamondsMLP = new ds.ml.MLPRegressor({
  hiddenLayers: [32, 32],
  activation: 'relu',
  learningRate: 1e-5,
  epochs: 20,
  batchSize: 512,
  verbose: false
});

diamondsMLP.fit({
  data: diamondsRecipePrepped.train.data,
  X: diamondsRecipePrepped.train.X,
  y: diamondsRecipePrepped.train.y
});

const diamondTestPredsRaw = diamondsMLP.predict({
  data: diamondsRecipePrepped.test.data,
  X: diamondsRecipePrepped.test.X
});
const diamondTestPreds = diamondTestPredsRaw.map((row) => row[0]);

const diamondTestActual = diamondsRecipePrepped.test.data.map((row) => row.price);

console.log('Test R²:', ds.ml.metrics.r2(diamondTestActual, diamondTestPreds).toFixed(3));
console.log('Test MAE:', ds.ml.metrics.mae(diamondTestActual, diamondTestPreds).toFixed(0), '$');
console.log('Test RMSE:', Math.sqrt(ds.ml.metrics.mse(diamondTestActual, diamondTestPreds)).toFixed(0), '$');

// %% [markdown]
/*
The R² computed on the test set is pretty high. The model has:
- MAE (Mean Absolute Error): Average prediction error in dollars
- RMSE (Root Mean Squared Error): Penalizes large errors more heavily

Both metrics show the model predicts diamond prices reasonably well given the features.
*/

// %% [markdown]
/*
ML models find patterns in data. Garbage in, garbage out—data quality matters more than model complexity!
*/

// %% [markdown]
/*
## Ensemble

Instead of relying on a single model, ensemble methods combine predictions from multiple models to improve accuracy and robustness (you might have read about ensemble methods from the previous section on clustering). This is the "wisdom of crowds" principle applied to machine learning, since different models make different mistakes. By combining them, we can reduce overfitting, improve generalization and capture different patterns. We do this with a `BranchPipeline`, which runs multiple models and combines their predictions
*/

// %% [javascript]

// Create ensemble of different classifiers
const ensembleClassifier = new ds.ml.BranchPipeline({
  branches: {
    knn: new ds.ml.KNNClassifier({ k: 10, weight: 'distance' }),
    tree: new ds.ml.DecisionTreeClassifier({ maxDepth: 10 }),
    forest: new ds.ml.RandomForestClassifier({ nTrees: 50, maxDepth: 10 })
  },
  combiner: 'vote'  // Majority voting for classification
});

// Fit all models (BranchPipeline expects array format)
ensembleClassifier.fit(
  penguinsTrainScaled.data.map(row =>
    penguinsFeatureFields.map(col => row[col])
  ),
  penguinsSplit.train.y  // encoded labels
);

// Predict with ensemble
const ensemblePredictions = ensembleClassifier.predict(
  penguinsTestScaled.data.map(row =>
    penguinsFeatureFields.map(col => row[col])
  )
);

// Decode predictions using the original split's encoder
const speciesEncoder = penguinsSplit.train.metadata.encoders.Species;
const ensembleLabels = ensemblePredictions.map(idx => speciesEncoder.decode(idx));

// Evaluate
console.log('Ensemble Accuracy:',
  ds.ml.metrics.accuracy(penguinsTrueSpecies, ensembleLabels).toFixed(3)
);

// %% [markdown]
/*
One advantage of ensembles is the confidence measurement, telling how much do the models agree.
*/

// %% [javascript]

// Check confidence for each prediction
const ensembleConfidence = ensembleClassifier.confidence(
  penguinsTestScaled.data.map(row =>
    penguinsFeatureFields.map(col => row[col])
  )
);

// Show samples with low confidence (models disagree)
console.log('Low confidence predictions (models disagree):');
ensembleConfidence.forEach((conf, i) => {
  if (conf < 0.8) {  // Less than 80% agreement
    console.log(`  Sample ${i}: ${ensembleLabels[i]} (confidence: ${(conf * 100).toFixed(0)}%)`);
  }
});

// Overall agreement
const avgConfidence = ensembleConfidence.reduce((a, b) => a + b) / ensembleConfidence.length;
console.log(`\nAverage ensemble confidence: ${(avgConfidence * 100).toFixed(1)}%`);

// %% [markdown]
/*
BranchPipeline supports different ways to combine predictions:

| Combiner | Use Case | How it works |
|----------|----------|--------------|
| `'vote'` | Classification | Majority vote wins |
| `'weighted_vote'` | Classification | Weighted majority (better models count more) |
| `'average'` | Regression | Average predictions |
| `'max'` / `'min'` | Regression | Take maximum/minimum prediction |
| Custom function | Any | Full control over combination logic |

For regression (predicting numbers),
*/

// %% [javascript]

// Ensemble for regression (diamond prices)
const ensembleRegressor = new ds.ml.BranchPipeline({
  branches: {
    knn: new ds.ml.KNNRegressor({ k: 10 }),
    tree: new ds.ml.DecisionTreeRegressor({ maxDepth: 15 }),
    forest: new ds.ml.RandomForestRegressor({ nTrees: 50, maxDepth: 15 })
  },
  combiner: 'average'  // Average predictions for regression
});

// Prepare data as arrays (models expect numeric arrays)
const XtrainDiamonds = diamondsRecipePrepped.train.data.map(row =>
  diamondsRecipePrepped.train.X.map(col => row[col])
);
const ytrainDiamonds = diamondsRecipePrepped.train.data.map(row => row.price);

const XtestDiamonds = diamondsRecipePrepped.test.data.map(row =>
  diamondsRecipePrepped.test.X.map(col => row[col])
);
const ytestDiamonds = diamondsRecipePrepped.test.data.map(row => row.price);

// Fit ensemble
ensembleRegressor.fit(XtrainDiamonds, ytrainDiamonds);

// Predict
const ensembleDiamondPreds = ensembleRegressor.predict(XtestDiamonds);

// Evaluate
console.log('Ensemble R²:', ds.ml.metrics.r2(ytestDiamonds, ensembleDiamondPreds).toFixed(3));
console.log('Ensemble MAE:', ds.ml.metrics.mae(ytestDiamonds, ensembleDiamondPreds).toFixed(0));

// Compare to single MLP
console.log('\nSingle MLP R²:', ds.ml.metrics.r2(diamondTestActual, diamondTestPreds).toFixed(3));
console.log('Single MLP MAE:', ds.ml.metrics.mae(diamondTestActual, diamondTestPreds).toFixed(0));
