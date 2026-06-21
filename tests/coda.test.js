import { describe, expect, it } from "vitest";
import { imputeMissing, compositionalOutliers, clr } from "../src/mva/composition.js";

describe("CoDA diagnostics", () => {
  describe("imputeMissing", () => {
    it("returns data unchanged when nothing is missing", () => {
      const X = [[1, 2, 3], [2, 2, 1]];
      expect(imputeMissing(X)).toEqual(X);
    });

    it("recovers a missing value on perfectly proportional data", () => {
      // All rows share the ratio 1:2:3, so the missing cell is determined.
      const X = [
        [2, NaN, 6], // true middle = 4
        [1, 2, 3],
        [3, 6, 9],
        [4, 8, 12],
      ];
      const out = imputeMissing(X);
      expect(out[0][1]).toBeCloseTo(4, 4);
      // strictly positive, observed entries preserved
      expect(out.flat().every((v) => v > 0)).toBe(true);
      expect(out[0][0]).toBe(2);
      expect(out[0][2]).toBe(6);
    });

    it("keeps the imputed sub-composition coherent (CLR ratios)", () => {
      const X = [
        [10, 20, 30, 40],
        [5, 10, 15, 20],
        [2, 4, NaN, 8], // true 6
        [1, 2, 3, 4],
      ];
      const out = imputeMissing(X);
      expect(out[2][2]).toBeCloseTo(6, 3);
    });
  });

  describe("compositionalOutliers", () => {
    it("flags a clear compositional outlier and not the bulk", () => {
      const bulk = Array.from({ length: 40 }, (_, i) => {
        const j = (i % 5) * 0.02;
        return [10 + j, 20 - j, 30 + j, 40 - j];
      });
      const outlier = [40, 5, 5, 5]; // very different composition
      const X = [...bulk, outlier];
      const res = compositionalOutliers(X, { alpha: 0.05 });

      expect(res.df).toBe(3); // D - 1
      expect(res.outliers[X.length - 1]).toBe(true);
      // the bulk should be mostly inliers
      const bulkFlagged = res.outliers.slice(0, bulk.length).filter(Boolean).length;
      expect(bulkFlagged).toBeLessThan(bulk.length * 0.2);
      // p-values are valid probabilities
      expect(res.pValues.every((p) => p >= 0 && p <= 1)).toBe(true);
      expect(res.distances[X.length - 1]).toBeGreaterThan(res.distances[0]);
    });

    it("uses a reference subpopulation for the centroid/covariance", () => {
      // Reference varies (mildly) in every component -> non-degenerate covariance.
      const ref = Array.from({ length: 24 }, (_, i) => {
        const a = ((i * 7) % 5) - 2, b = ((i * 3) % 5) - 2, c = ((i * 11) % 5) - 2;
        return [10 + a * 0.3, 20 + b * 0.3, 30 + c * 0.3, 40 - (a + b + c) * 0.3];
      });
      const probe = [[10, 20, 30, 40], [80, 5, 5, 10]];
      const X = [...ref, ...probe];
      const reference = X.map((_, i) => i < ref.length);
      const res = compositionalOutliers(X, { reference, alpha: 0.05 });
      // the probe near the reference centroid is an inlier; the far one is not
      expect(res.outliers[ref.length]).toBe(false);
      expect(res.outliers[ref.length + 1]).toBe(true);
    });
  });
});
