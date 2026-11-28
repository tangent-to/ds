// ---
// title: A neural regressor, from folds to a saved model
// id: ds-neural-regression
// ---

// %% [markdown]
/*
# A neural regressor, from folds to a saved model

A neural network is one more regressor. It earns its place the way any model does: by predicting rows it has not seen better than a simpler model, on the same rows, with the same preprocessing. This chapter takes a linear model and a small multi-layer perceptron (MLP) through that comparison, then carries the winner all the way to something an application can use: a saved file that predicts from raw records.

1. Load a table of named records.
2. Build the two models as pipelines, so scaling is learned inside each training fold.
3. Score both on identical folds.
4. Refit the better one on all the data and save it.
5. Reload it, as a fresh page would, and predict from records whose fields arrive in any order.
*/

// %% [javascript]

import ds from '../../dist/index.js';
import * as Plot from 'https://esm.sh/@observablehq/plot';

const carsResponse = await fetch('https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json');
const carsRaw = await carsResponse.json();

// %% [markdown]
/*
## 1. The data

The cars data set records the fuel consumption of 406 cars from the 1970s and early 1980s, with their engine and body characteristics. We predict miles per gallon from five numeric features, and keep only the cars where all six are recorded. Each row stays a named record: the model is told which fields to read, so nothing depends on the order of the keys.
*/

// %% [javascript]

const carFeatures = ['Cylinders', 'Displacement', 'Horsepower', 'Weight_in_lbs', 'Acceleration'];
const cars = carsRaw.filter((row) =>
  row.Miles_per_Gallon != null && carFeatures.every((field) => row[field] != null));
const mpg = cars.map((row) => row.Miles_per_Gallon);

console.log(`${cars.length} of ${carsRaw.length} cars are complete`);
cars.slice(0, 5)

// %% [markdown]
/*
Fuel consumption falls with weight, but not along a straight line: the drop flattens for heavy cars. A linear model has to average over that curve; an MLP can bend with it. Whether that flexibility pays off on new cars, rather than on the ones it learned from, is what the folds below measure.
*/

// %% [javascript]

Plot.plot({
  x: { label: 'Weight (lbs)' },
  y: { label: 'Miles per gallon' },
  marks: [
    Plot.dot(cars, { x: 'Weight_in_lbs', y: 'Miles_per_Gallon', fill: 'Cylinders', r: 3 }),
    Plot.linearRegressionY(cars, { x: 'Weight_in_lbs', y: 'Miles_per_Gallon' }),
  ],
  color: { legend: true, type: 'ordinal' },
})

// %% [markdown]
/*
## 2. Two pipelines

Each model is a `Pipeline`: a `StandardScaler` followed by the estimator. Scaling is part of the model, not a step done to the data beforehand. That matters twice. In cross-validation, the means and standard deviations are learned from each training fold alone, so the test fold never leaks into them. And when the model is saved, the scaling is saved with it.

The `columns` option names the features, in order. The pipeline then reads those fields from each record, ignores any other field, and refuses a record where one is missing rather than quietly dropping it.

The MLP has one hidden layer of 16 `tanh` units. On a few hundred rows, L-BFGS on the full batch is the right optimizer: each of its 200 iterations uses the exact gradient over every training row, where Adam would need many epochs of mini-batches; `normalizeY` (on by default) standardizes the target for training and maps predictions back to miles per gallon. The seed fixes the starting weights.
*/

// %% [javascript]

const { Pipeline, MLPRegressor, preprocessing, metrics, validation } = ds.ml;

const makeLinear = () => new Pipeline(
  [new preprocessing.StandardScaler(), new ds.stats.GLM({ family: 'gaussian' })],
  { columns: carFeatures },
);
const makeMLP = () => new Pipeline(
  [
    new preprocessing.StandardScaler(),
    new MLPRegressor({
      layerSizes: [carFeatures.length, 16, 1],
      activation: 'tanh',
      optimizer: 'lbfgs',
      epochs: 200,
      seed: 1,
    }),
  ],
  { columns: carFeatures },
);
const candidates = { linear: makeLinear, MLP: makeMLP };

// %% [markdown]
/*
## 3. Identical folds

Both models are scored on the same five folds. The data are shuffled once, with a seed so the folds are the same on every run, then cut into five; each fold is held out in turn while the models learn from the other four. Because the folds are shared, the comparison is paired: on each fold both models faced exactly the same test cars.

For every held-out car we keep the prediction of each model, which gives the scores per fold and a picture of the errors.
*/

// %% [javascript]

ds.ml.utils.setSeed(2024);
const carFolds = validation.kFold(cars, mpg, 5, true);

const foldScores = [];
const outOfFold = [];
carFolds.forEach(({ trainIndices, testIndices }, f) => {
  const trainRows = trainIndices.map((i) => cars[i]);
  const trainMpg = trainIndices.map((i) => mpg[i]);
  const testRows = testIndices.map((i) => cars[i]);
  const testMpg = testIndices.map((i) => mpg[i]);
  for (const [model, make] of Object.entries(candidates)) {
    const predicted = make().fit(trainRows, trainMpg).predict(testRows);
    foldScores.push({
      fold: f + 1,
      model,
      RMSE: metrics.rmse(testMpg, predicted),
      R2: metrics.r2(testMpg, predicted),
    });
    testIndices.forEach((i, j) => outOfFold.push({ model, observed: mpg[i], predicted: predicted[j] }));
  }
});

foldScores

// %% [markdown]
/*
A line joins the two scores of each fold. When every line slopes the same way, the difference is not an accident of one lucky split.
*/

// %% [javascript]

Plot.plot({
  x: { label: null, domain: Object.keys(candidates) },
  y: { label: 'RMSE on the held-out fold (mpg)', grid: true },
  marks: [
    Plot.line(foldScores, { x: 'model', y: 'RMSE', z: 'fold', stroke: '#bbb' }),
    Plot.dot(foldScores, { x: 'model', y: 'RMSE', fill: 'model', r: 5 }),
    Plot.text(foldScores.filter((d) => d.model === 'MLP'), { x: 'model', y: 'RMSE', text: (d) => `fold ${d.fold}`, dx: 12, textAnchor: 'start' }),
  ],
})

// %% [javascript]

const summary = Object.keys(candidates).map((model) => {
  const scores = foldScores.filter((d) => d.model === model);
  const mean = (key) => scores.reduce((s, d) => s + d[key], 0) / scores.length;
  return { model, 'mean RMSE': mean('RMSE'), 'mean R2': mean('R2') };
});
const bestModel = summary.reduce((a, b) => (b['mean RMSE'] < a['mean RMSE'] ? b : a)).model;

console.log(`Lower mean RMSE: ${bestModel}`);
summary

// %% [markdown]
/*
The out-of-fold predictions show where the gain comes from. A perfect model would put every car on the dashed diagonal. The cloud of the linear model is straight where the data are curved: it under-predicts the most economical cars, the ones on the flat end of the curve, and over-predicts the middle of the range to compensate. The MLP follows the curve further, though it too falls short on the most economical cars. The table after the chart averages the error, predicted minus observed, in three bands of fuel economy: a negative mean error is an under-prediction.
*/

// %% [javascript] #wide

Plot.plot({
  width,
  grid: true,
  x: { label: 'Observed (mpg)' },
  y: { label: 'Predicted on the held-out fold (mpg)' },
  fx: { label: null },
  marks: [
    Plot.line([[9, 9], [47, 47]], { stroke: '#999', strokeDasharray: '4 3' }),
    Plot.dot(outOfFold, { x: 'observed', y: 'predicted', fx: 'model', r: 2.5, fill: 'model' }),
  ],
})

// %% [javascript]

const bands = [[0, 15], [15, 30], [30, Infinity]];
bands.map(([low, high]) => {
  const inBand = (model) => outOfFold.filter((d) => d.model === model && d.observed >= low && d.observed < high);
  const meanError = (model) => {
    const rows = inBand(model);
    return rows.reduce((s, d) => s + d.predicted - d.observed, 0) / rows.length;
  };
  return {
    'observed mpg': high === Infinity ? `${low} and over` : `${low} to ${high}`,
    cars: inBand('linear').length,
    'linear: mean error': meanError('linear'),
    'MLP: mean error': meanError('MLP'),
  };
})

// %% [markdown]
/*
## 4. Refit and save

Cross-validation chose a recipe, not a fitted model. The model to keep is that recipe refit on every car, which is the most data it will ever learn from.

`save()` writes the whole pipeline as JSON text: the feature names and their order, the scaler's means and standard deviations, and the network with its weights and the target scaling. That text is everything a page needs to predict; there is nothing else to ship.
*/

// %% [javascript]

const finalModel = candidates[bestModel]().fit(cars, mpg);
const savedModel = finalModel.save();

console.log(`${bestModel} pipeline saved: ${(savedModel.length / 1024).toFixed(1)} kB of JSON`);
JSON.parse(savedModel).data.columns

// %% [markdown]
/*
In an application, that text goes to a file or to storage. Note's `save` writes it next to the notebook: into the folder on disk when the notebook is served by `note serve`, into the notebook's own browser storage otherwise (the Storage panel lists it and can download it).
*/

// %% [javascript]

const modelFile = 'mpg-model.json';
const written = await save(modelFile, savedModel);
written

// %% [markdown]
/*
## 5. Reload and predict from records

`Pipeline.load` rebuilds the model from the file alone, as a fresh page would. (Where the browser offers no storage, `save` falls back to a download and the text is read from memory instead.) The new cars below are records the way an application might receive them: fields in a different order, an extra `id` that the model never saw. The pipeline reads the five features it was trained on, by name.
*/

// %% [javascript]

const modelText = written.downloaded ? savedModel : await FileAttachment(modelFile).text();
const reloaded = ds.ml.Pipeline.load(modelText);

const newCars = [
  { id: 'compact', Weight_in_lbs: 2200, Acceleration: 16, Horsepower: 70, Cylinders: 4, Displacement: 98 },
  { id: 'sedan', Horsepower: 105, Cylinders: 6, Weight_in_lbs: 3100, Displacement: 200, Acceleration: 15.5 },
  { id: 'wagon', Displacement: 350, Acceleration: 12, Cylinders: 8, Horsepower: 165, Weight_in_lbs: 4300 },
];
const reloadedMpg = reloaded.predict(newCars);
const sameAsBefore = finalModel.predict(newCars).every((v, i) => v === reloadedMpg[i]);

console.log(`Reloaded predictions identical to the original model: ${sameAsBefore}`);
newCars.map((car, i) => ({ id: car.id, 'predicted mpg': reloadedMpg[i] }))

// %% [markdown]
/*
A record without one of the features is refused, with the name of the missing field, instead of being dropped from the output or filled with a guess. If a value can be missing in practice, impute it in a step of the pipeline (see the chapter on outliers and imputation) so the choice is saved along with the model.
*/

// %% [javascript]

try {
  reloaded.predict([{ id: 'incomplete', Cylinders: 4, Displacement: 98, Horsepower: 70, Acceleration: 16 }]);
} catch (error) {
  console.log(`Refused: ${error.message}`);
}

// %% [markdown]
/*
## Where to go from here

- `MLPRegressor` runs on [tangent/nn](https://github.com/tangent-to/nn). For branches, dropout that learns its rate, or a loss other than squared error, build the network there and wrap it the same way.
- Tune the hidden layer size or the number of iterations with `GridSearchCV`, on the same folds, as in the machine learning chapter.
- A model that predicts a mean says nothing about its uncertainty. The Gaussian process regressor and nn's `gaussianNLL` loss predict a spread as well.
*/
