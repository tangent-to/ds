import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import nn from '@tangent.to/nn';
import { describe, expect, it } from 'vitest';
import { ml, stats } from '../src/index.js';

const { Pipeline, MLPRegressor, preprocessing: { StandardScaler }, metrics, validation } = ml;
const columns = ['girth', 'height'];
const X = [[8, 70], [9, 63], [11, 79], [12, 71], [13, 75], [15, 68]];
const y = [10, 12, 22, 26, 31, 38];
const named = (rows) => rows.map(([girth, height]) => ({ girth, height }));
// The test rows arrive out of order and contain an unused field.
const newRows = [{ height: 72, id: 'A', girth: 10 }, { height: 80, id: 'B', girth: 14 }];
const settings = {
  layerSizes: [2, 2, 1],
  activation: 'tanh',
  optimizer: 'sgd',
  learningRate: 0.04,
  epochs: 1,
  batchSize: X.length,
  normalizeY: true,
  seed: 17,
};
const keys = ['dense_1/W', 'dense_1/b', 'dense_2/W', 'dense_2/b'];
const initialModel = () => {
  const input = nn.input(2);
  const hidden = nn.dense(2, { activation: settings.activation })(input);
  return nn.model(input, nn.dense(1)(hidden), { seed: settings.seed });
};
const parameters = (model) => {
  const params = model.getParams();
  return keys.map((key) => {
    const values = params[key].data;
    expect(values).toBeInstanceOf(Float64Array);
    return Array.from(values);
  });
};
const makePipeline = (model) => new Pipeline([new StandardScaler(), model], { columns });
const fitLinear = (train, target) => makePipeline(new stats.GLM({ family: 'gaussian' })).fit(train, target);
const fitNeural = (train, target) => makePipeline(new MLPRegressor(settings)).fit(train, target);

function reloadInFreshProcess(pipelines, rows) {
  const loader = `
    import { readFileSync } from 'node:fs';
    import { ml } from ${JSON.stringify(new URL('../src/index.js', import.meta.url).href)};
    const { pipelines, rows } = JSON.parse(readFileSync(0, 'utf8'));
    console.log(JSON.stringify(pipelines.map((saved) => ml.Pipeline.load(saved).predict(rows))));
  `;
  return JSON.parse(execFileSync(process.execPath, ['--input-type=module', '--eval', loader], {
    input: JSON.stringify({ pipelines: pipelines.map((p) => p.save()), rows }),
    encoding: 'utf8',
  }));
}

describe('fitted regression pipelines', () => {
  it('reloads linear and neural pipelines in a fresh process and predicts from named rows', () => {
    const pipelines = [fitLinear(named(X), y), fitNeural(named(X), y)];
    const expected = pipelines.map((p) => p.predict(newRows));
    expect(reloadInFreshProcess(pipelines, newRows)).toEqual(expected);
  });

  it('round-trips matrix inputs and rejects unsupported steps and missing named features', () => {
    const matrix = new Pipeline([new StandardScaler(), new MLPRegressor(settings)]).fit(X, y);
    expect(Pipeline.load(matrix.save()).predict(X)).toEqual(matrix.predict(X));

    const pipeline = Pipeline.load(fitNeural(named(X), y).save());
    expect(() => pipeline.predict([{ girth: 10, height: null }])).toThrow(/height/);
    expect(() => Pipeline.fromJSON({ ...pipeline.toJSON(), steps: [{ __class__: 'constructor' }] }))
      .toThrow(/cannot restore step 0/);
    const custom = new Pipeline([{ fit() {}, predict() {} }]).fit(X, y);
    expect(() => custom.save()).toThrow(/cannot serialize step 0/);
  });
});

function compareNumbers(actual, expected, label) {
  if (Array.isArray(actual)) {
    expect(expected, label).toHaveLength(actual.length);
    actual.forEach((value, i) => compareNumbers(value, expected[i], `${label}[${i}]`));
  } else {
    expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(1e-10 + 1e-10 * Math.abs(expected));
  }
}

const references = [
  ['R', 'Rscript', '../tests_compare-to-R/tabular_regression.R'],
  ['Python', 'python3', '../tests_compare-to-python/tabular_regression.py'],
];

it.skipIf(process.env.TANGENT_REGRESSION_REFERENCES !== '1').each(references)(
  'matches the %s Float64 reference on shared regression folds and after reload',
  (_language, interpreter, script) => {
    const folds = validation.kFold(X, y, 3);
    const full = { trainIndices: X.map((_, i) => i), testIndices: [] };
    const initial = initialModel();
    const weights = parameters(initial);
    const results = [...folds, full].map(({ trainIndices, testIndices }) => {
      const train = named(trainIndices.map((i) => X[i]));
      const target = trainIndices.map((i) => y[i]);
      const test = testIndices.length ? named(testIndices.map((i) => X[i])) : newRows;
      const linear = fitLinear(train, target);
      const neural = fitNeural(train, target);
      const [scaler] = neural.steps;
      const estimator = neural.getFinalEstimator();
      const scaledTrain = scaler.transform(trainIndices.map((i) => X[i]));
      const [linearPrediction, neuralPrediction] = testIndices.length
        ? [linear.predict(test), neural.predict(test)]
        : reloadInFreshProcess([linear, neural], test);
      const actual = {
        means: scaler.means,
        scales: scaler.stds,
        scaledTrain,
        scaledTest: scaler.transform(test.map((row) => columns.map((column) => row[column]))),
        initialPrediction: initial.predict(scaledTrain),
        loss: estimator.history.loss[0],
        targetMean: estimator.model.yStats.mean[0],
        targetScale: estimator.model.yStats.sd[0],
        updatedWeights: parameters(estimator.model),
        linearPrediction,
        neuralPrediction,
      };
      if (testIndices.length) {
        const testTarget = testIndices.map((i) => y[i]);
        actual.rmse = [metrics.rmse(testTarget, linearPrediction), metrics.rmse(testTarget, neuralPrediction)];
      }
      return actual;
    });
    const ref = JSON.parse(execFileSync(interpreter, [fileURLToPath(new URL(script, import.meta.url))], {
      input: JSON.stringify({
        X, y, folds, weights, learningRate: settings.learningRate,
        newX: newRows.map((row) => columns.map((column) => row[column])),
      }),
      encoding: 'utf8',
      timeout: 30_000,
    }));
    expect(ref).toHaveLength(results.length);
    results.forEach((result, i) => {
      for (const [key, value] of Object.entries(result)) {
        compareNumbers(value, ref[i][key], `fit ${i + 1} ${key}`);
      }
    });
  },
);
