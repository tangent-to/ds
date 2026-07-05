// ---
// title: API Patterns in tangent/ds
// id: ds-api
// ---

// %% [markdown]
/*
# API Patterns in tangent/ds

The tangent/ds package supports three different API patterns, each designed for different use cases and preferences. This notebook explains when and how to use each pattern.

## The three API patterns

1. **Array API**: Pure numeric arrays, inspired by NumPy/SciPy
2. **Declarative Table API**: Column-based data with metadata, inspired by R and pandas
3. **Recipe API**: Chainable preprocessing workflows with full inspection

All three patterns are available across the package: MVA (PCA, LDA, RDA), Stats (GLM), and ML (preprocessing, estimators).
*/

// %% [javascript]

// our base library
import ds from '../../dist/index.js';

// data
const penguinsResponse = await fetch(
  'https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json',
);
const penguinsDataRaw = await penguinsResponse.json();

// There is a data with a . instead of null...
const penguinsData = penguinsDataRaw
  .map(row => row.Sex === '.' ? { ...row, Sex: null } : row)
  .filter(row => row.Sex);

console.table(penguinsData.slice(0, 5));

// %% [markdown]
/*
## Array API

The Array API works with pure numeric matrices (arrays of arrays). This is the most direct approach for users familiar with NumPy, MATLAB, or scikit-learn. The array API is fed by arrays, and also returns arrays. You might prefer to use it when

- you already have numeric data in array format,
- you want minimal abstraction and maximum performance,
- you want to remove missing values yourself,
- you're translating code from Python/NumPy/scikit-learn or
- you don't need column names or metadata.

Here is an example to perform a PCA with the array API.
*/

// %% [javascript]

const numericColumns = [
  "Beak Length (mm)",
  "Beak Depth (mm)",
  "Flipper Length (mm)",
  "Body Mass (g)"
];

// Keep Array API input aligned with Table API (naOmit=true drops rows with missing values).
const penguinsArray = penguinsData
  .filter(d => numericColumns.every(col => d[col] != null))
  .map(d => numericColumns.map(col => d[col]));

const pca = ds.mva.pca.fit(penguinsArray, { scale: true, center: true });
const scores = ds.mva.pca.transform(pca, penguinsArray);
console.log(scores.slice(0, 3))

// %% [markdown]
/*
Let's see how scaling would work with the array API.
*/

// %% [javascript]

const penguinsScaler = new ds.ml.preprocessing.StandardScaler().fit(penguinsArray);
const penguinsScaled = penguinsScaler.transform(penguinsArray);
console.log('\nScaled data (first sample):', penguinsScaled[0]);

// %% [markdown]
/*
Machine learning can also be approached with arrays.
*/

// %% [javascript]

const species = penguinsData.map(d => d.Species);

// trainTestSplit with Array API
const penguinsSplit = ds.ml.validation.trainTestSplit(
  penguinsArray,
  species,
  { ratio: 0.7, shuffle: true, seed: 42 }
);

// Fit scaler on training data only
const penguinsScalerML = new ds.ml.preprocessing.StandardScaler()
  .fit(penguinsSplit.XTrain);

// Transform train and test separately
const XTrainScaled = penguinsScalerML.transform(penguinsSplit.XTrain);
const XTestScaled = penguinsScalerML.transform(penguinsSplit.XTest);

// Fit classifier
const knn = new ds.ml.KNNClassifier({ k: 10 });
knn.fit(XTrainScaled, penguinsSplit.yTrain);

// Predict
const predictions = knn.predict(XTestScaled);
undefined;

// %% [markdown]
/*
## Declarative table API

The Table API works with structured data (arrays of objects) and column selectors. This approach preserves column names, handles missing values, and maintains metadata throughout the analysis. It was designed in the perspective where

- your data is already in table format (CSV, JSON, database),
- you want to preserve column names and metadata,
- you need automatic handling of mixed data types,
- you prefer longer but declarative, self-documenting code, or
- you're familiar with more tidy approaches in R, pandas, or SQL.

The table API is fed with data (Arrays of objects or Arquero), and outputs objects with data, column names, and metadata. It automates common tasks.

Let's try it for PCA.
*/

// %% [javascript]

const pcaTable = ds.mva.pca.fit({
  data: penguinsData,
  columns: numericColumns,
  scale: true,
  center: true
});
console.table(pcaTable.scores.slice(0, 3));

// %% [markdown]
/*
With the same scaling operation as before...
*/

// %% [javascript]

const scalerTable = new ds.ml.preprocessing.StandardScaler();
scalerTable.fit({
  data: penguinsData,
  columns: numericColumns,

});
const scaledTable = scalerTable.transform({
  data: penguinsData,
  columns: numericColumns,
});
console.log('\nScaled data (first sample):', scaledTable.data[0]);

// %% [javascript]

// Example 4: End-to-end with Table API

// Split preserves all information
const tableSplit = ds.ml.validation.trainTestSplit(
  {
    data: penguinsData,
    X: numericColumns,
    y: 'Species'
  },
  { ratio: 0.7, shuffle: true, seed: 42 }
);

// Fit scaler on training data
const tableScaler = new ds.ml.preprocessing.StandardScaler()
  .fit({
    data: tableSplit.train.data,
    columns: numericColumns
  });

// Transform both train and test (encoders pass through)
const tableTrainScaled = tableScaler.transform({
  data: tableSplit.train.data,
  columns: numericColumns,
  encoders: tableSplit.train.metadata.encoders
});

const tableTestScaled = tableScaler.transform({
  data: tableSplit.test.data,
  columns: numericColumns,
  encoders: tableSplit.train.metadata.encoders  // Use TRAIN encoders
});

// Fit model (encoders ensure consistent label encoding)
const tableKnn = new ds.ml.KNNClassifier({ k: 3 }).fit({
  data: tableTrainScaled.data,
  X: numericColumns,
  y: 'Species',
  encoders: tableTrainScaled.metadata.encoders
});

// Predict
const tablePreds = tableKnn.predict({
  data: tableTestScaled.data,
  X: numericColumns,
  encoders: tableTestScaled.metadata.encoders
});

console.log('Predictions:', tablePreds);

// %% [markdown]
/*
## Recipe API

The recipe API provides a chainable, declarative way to define preprocessing workflows. Akin to Posit R's recipe approach, the recipe API allows full inspection of intermediate results while ensuring transformers are correctly fitted and applied. Use it when

- you have complex preprocessing with multiple steps,
- you want to inspect intermediate transformations,
- you need to apply the same preprocessing to new data,
- you want to avoid manual encoder/metadata passing, or
- you prefer verbose but readable, self-documenting workflows.

The recipe API works with chainable methods defining preprocessing steps, with `prep()` executing all steps and returns inspectable results (important for intermediate verifications), and with `bake()` applying fitted transformers to new data. Transformers are stored and reusable.
*/

// %% [javascript]

const numericFeatures = ["Beak Length (mm)", "Beak Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"];
const preprocessRecipe = ds.ml.recipe({
  data: penguinsData,
    X: [ ... numericFeatures, 'Island', 'Sex'],
    y: 'Species'
  })
  .oneHot(['Island', 'Sex'], { dropFirst: true })
  .scale(numericFeatures)
  .split({ ratio: 0.7, shuffle: true, seed: 42 });

const prepped = preprocessRecipe.prep()

const knnRecipe = new ds.ml.KNNClassifier({ k: 3 }).fit({
  data: prepped.train.data,
  X: prepped.train.X,
  y: prepped.train.y,
  encoders: prepped.train.metadata.encoders
});

const predictionsRecipe = knnRecipe.predict({
  data: prepped.test.data,
  X: prepped.test.X,
  encoders: prepped.test.metadata.encoders
});

// Calculate accuracy
const actual = prepped.test.data.map(d => d.Species);
const correct = predictionsRecipe.filter((p, i) => p === actual[i]).length;
console.log('Accuracy:', (correct / predictionsRecipe.length * 100).toFixed(1) + '%');

// %% [javascript]

// Example 4: Apply Recipe to New Data
const newPenguin = [
  {
    "Species": "Unknown",
    Island: "Biscoe", Sex: "FEMALE",
    "Beak Length (mm)": 45.0, "Beak Depth (mm)": 15.0, "Flipper Length (mm)": 210, "Body Mass (g)": 4800
  }
];

const newPrepped = preprocessRecipe.bake(newPenguin);// bake() applies all fitted transformers
const newPrediction = model.predict({
  data: newPrepped.data,
  X: newPrepped.X
});
console.log('\nPredicted species:', newPrediction[0]);
