import { describe, expect, it } from "vitest";
import * as explain from "../src/ml/explain.js";
import { DecisionTreeRegressor } from "../src/ml/estimators/DecisionTree.js";
import { RandomForestRegressor } from "../src/ml/estimators/RandomForest.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// A known linear model: f(x) = 2 + 3*x0 - 1*x1 + 0.5*x2
const COEF = [3, -1, 0.5];
const INTERCEPT = 2;
const linearPredict = (rows) =>
  rows.map((r) => INTERCEPT + r[0] * COEF[0] + r[1] * COEF[1] + r[2] * COEF[2]);

// Background sample with simple, distinct column means.
const background = [
  [0, 0, 0],
  [2, 2, 2],
  [4, 0, 8],
  [0, 4, 4],
  [1, 3, 2],
  [3, 1, 6],
];
const bgMean = [0, 1, 2].map(
  (j) => background.reduce((s, r) => s + r[j], 0) / background.length,
);

describe("SHAP explainers", () => {
  describe("KernelExplainer", () => {
    it("recovers exact Shapley values for a linear model", () => {
      const ex = new explain.KernelExplainer({
        predict: linearPredict,
        background,
        featureNames: ["a", "b", "c"],
      });
      const x = [5, 1, 3];
      const { values, baseValue } = ex.shapValues([x]);
      const phi = values[0];

      // For a linear model with an interventional background, the exact SHAP
      // value of feature j is coef_j * (x_j - E[x_j]).
      for (let j = 0; j < 3; j++) {
        expect(phi[j]).toBeCloseTo(COEF[j] * (x[j] - bgMean[j]), 6);
      }
      // base value == mean prediction over background
      expect(baseValue).toBeCloseTo(_mean(linearPredict(background)), 6);
    });

    it("is additive: baseValue + sum(phi) == f(x)", () => {
      const ex = new explain.KernelExplainer({ predict: linearPredict, background });
      const X = [
        [5, 1, 3],
        [-2, 4, 0],
      ];
      const { values, baseValue } = ex.shapValues(X);
      const fx = linearPredict(X);
      values.forEach((phi, i) => {
        const sum = baseValue + phi.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(fx[i], 6);
      });
    });

    it("works via the sampling path for higher dimensions (additivity)", () => {
      const M = 18;
      const coef = Array.from({ length: M }, (_, j) => (j % 3) - 1);
      const predict = (rows) =>
        rows.map((r) => r.reduce((s, v, j) => s + v * coef[j], 0));
      const rand = () => 0.5;
      const bg = Array.from({ length: 10 }, (_, i) =>
        Array.from({ length: M }, (_, j) => (i + j) % 5),
      );
      const ex = new explain.KernelExplainer({ predict, background: bg });
      const x = Array.from({ length: M }, (_, j) => j);
      const { values, baseValue } = ex.shapValues([x], { nSamples: 4000, seed: 1 });
      const sum = baseValue + values[0].reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(predict([x])[0], 4);
    });
  });

  describe("TreeExplainer", () => {
    // Small deterministic regression dataset.
    const X = [
      [1, 0], [2, 0], [3, 1], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1],
      [1, 1], [2, 1], [3, 0], [4, 0], [5, 1], [6, 0], [7, 1], [8, 0],
    ];
    const y = X.map((r) => 10 + 5 * r[0] + 8 * r[1]);

    it("is additive for a single decision tree", () => {
      const tree = new DecisionTreeRegressor({ maxDepth: 4 });
      tree.fit(X, y);
      const ex = new explain.TreeExplainer({ model: tree });
      const { values, baseValue } = ex.shapValues(X);
      const preds = tree.predict(X);
      values.forEach((phi, i) => {
        const sum = baseValue + phi.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(preds[i], 6);
      });
    });

    it("baseValue equals the tree's coverage-weighted mean output", () => {
      const tree = new DecisionTreeRegressor({ maxDepth: 4 });
      tree.fit(X, y);
      const ex = new explain.TreeExplainer({ model: tree });
      // On the training set, mean prediction == coverage-weighted leaf mean.
      const meanPred = _mean(tree.predict(X));
      expect(ex.expectedValue).toBeCloseTo(meanPred, 6);
    });

    it("is additive for a random forest", () => {
      const forest = new RandomForestRegressor({ nEstimators: 8, maxDepth: 4, random: 42 });
      forest.fit(X, y);
      const ex = new explain.TreeExplainer({ model: forest });
      const { values, baseValue } = ex.shapValues(X);
      const preds = forest.predict(X);
      values.forEach((phi, i) => {
        const sum = baseValue + phi.reduce((a, b) => a + b, 0);
        // Path-dependent forest SHAP is additive w.r.t. the forest's own
        // expected value; allow a little slack for averaging.
        expect(sum).toBeCloseTo(preds[i], 4);
      });
    });
  });

  describe("PermutationExplainer", () => {
    it("is additive for a linear model", () => {
      const ex = new explain.PermutationExplainer({ predict: linearPredict, background });
      const X = [[5, 1, 3], [-2, 4, 0]];
      const { values, baseValue } = ex.shapValues(X, { nPermutations: 32, seed: 3 });
      const fx = linearPredict(X);
      values.forEach((phi, i) => {
        const sum = baseValue + phi.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(fx[i], 6);
      });
    });

    it("approximately matches KernelSHAP on a linear model", () => {
      const x = [5, 1, 3];
      const kern = new explain.KernelExplainer({ predict: linearPredict, background })
        .shapValues([x]).values[0];
      const perm = new explain.PermutationExplainer({ predict: linearPredict, background })
        .shapValues([x], { nPermutations: 200, seed: 7 }).values[0];
      for (let j = 0; j < 3; j++) expect(perm[j]).toBeCloseTo(kern[j], 4);
    });
  });

  describe("plotting helpers", () => {
    it("importanceData returns sorted mean(|shap|) per feature", () => {
      const res = {
        values: [
          [3, -1, 0.5],
          [-3, 1, -0.5],
        ],
        featureNames: ["a", "b", "c"],
      };
      const imp = explain.importanceData(res);
      expect(imp.map((d) => d.feature)).toEqual(["a", "b", "c"]);
      expect(imp[0].importance).toBeCloseTo(3, 6);
      expect(imp[1].importance).toBeCloseTo(1, 6);
      expect(imp[2].importance).toBeCloseTo(0.5, 6);
    });

    it("summaryData returns one tidy row per (instance, feature)", () => {
      const res = { values: [[3, -1, 0.5]], featureNames: ["a", "b", "c"] };
      const rows = explain.summaryData(res, [[5, 1, 3]]);
      expect(rows).toHaveLength(3);
      expect(rows[0]).toMatchObject({ instance: 0, feature: "a", shap: 3, featureValue: 5 });
    });
  });
});

function _mean(a) {
  return a.reduce((s, v) => s + v, 0) / a.length;
}
