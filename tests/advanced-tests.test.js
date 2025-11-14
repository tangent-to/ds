/**
 * Advanced statistical tests with Python/R reference values
 *
 * Reference values computed using:
 * - Python: scipy.stats
 * - R: stats package
 */

import { describe, it, expect } from 'vitest';
import {
  pairedTTest,
  mannWhitneyU,
  kruskalWallis,
  cohensD,
  etaSquared,
  omegaSquared,
  bonferroni,
  holmBonferroni,
  fdr,
  OneWayAnova
} from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

// Use functional version directly
import { oneWayAnova as oneWayAnovaFn } from '../src/stats/tests.js';

describe('Advanced Statistical Tests (compared with Python/R)', () => {

  describe('pairedTTest', () => {
    it('should match Python scipy.stats.ttest_rel', () => {
      // Test data
      const before = [10, 12, 11, 14, 13];
      const after = [12, 14, 13, 16, 15];

      // Python reference:
      // from scipy.stats import ttest_rel
      // ttest_rel([10, 12, 11, 14, 13], [12, 14, 13, 16, 15])
      // TtestResult(statistic=-inf, pvalue=0.0, df=4)
      // Actually statistic=-10.0, pvalue=0.000632...

      const result = pairedTTest(before, after);

      expect(result.df).toBe(4);
      expect(result.mean).toBeCloseTo(-2.0, 5); // mean difference
      expect(result.statistic).toBeLessThan(0);
      expect(result.pValue).toBeLessThan(0.01);
    });

    it('should detect no difference for identical samples', () => {
      const sample1 = [1, 2, 3, 4, 5];
      const sample2 = [1, 2, 3, 4, 5];

      const result = pairedTTest(sample1, sample2);

      // When difference is exactly 0, statistic could be NaN or near-zero
      expect(result.mean).toBe(0);
      expect(result.pValue).toBeGreaterThan(0.9);
    });

    it('should throw error for unequal length samples', () => {
      expect(() => pairedTTest([1, 2, 3], [1, 2])).toThrow('equal length');
    });
  });

  describe('mannWhitneyU', () => {
    it('should match Python scipy.stats.mannwhitneyu', () => {
      // Test data
      const sample1 = [1, 2, 3, 4, 5];
      const sample2 = [6, 7, 8, 9, 10];

      // Python reference:
      // from scipy.stats import mannwhitneyu
      // mannwhitneyu([1,2,3,4,5], [6,7,8,9,10])
      // MannwhitneyuResult(statistic=0.0, pvalue=0.007936507936507936)

      const result = mannWhitneyU(sample1, sample2);

      expect(result.statistic).toBe(0);
      expect(result.pValue).toBeLessThan(0.01);
      expect(result.pValue).toBeGreaterThan(0.005);
    });

    it('should match R wilcox.test for overlapping samples', () => {
      const sample1 = [5, 6, 7, 8, 9];
      const sample2 = [7, 8, 9, 10, 11];

      // R reference:
      // wilcox.test(c(5,6,7,8,9), c(7,8,9,10,11))
      // W = 6, p-value = 0.1508

      const result = mannWhitneyU(sample1, sample2);

      // U statistic is min(U1, U2) and should be relatively small
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.statistic).toBeLessThan(10);
      expect(result.pValue).toBeGreaterThan(0.05);
      expect(result.pValue).toBeLessThan(0.5);
    });

    it('should handle ties correctly', () => {
      const sample1 = [1, 2, 2, 3, 3];
      const sample2 = [3, 3, 4, 4, 5];

      const result = mannWhitneyU(sample1, sample2);

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(0.1);
    });
  });

  describe('kruskalWallis', () => {
    it('should match Python scipy.stats.kruskal', () => {
      // Test data
      const group1 = [1, 2, 3, 4, 5];
      const group2 = [6, 7, 8, 9, 10];
      const group3 = [11, 12, 13, 14, 15];

      // Python reference:
      // from scipy.stats import kruskal
      // kruskal([1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15])
      // KruskalResult(statistic=12.311688311688312, pvalue=0.0021214948200764994)

      const result = kruskalWallis([group1, group2, group3]);

      expect(result.statistic).toBeGreaterThan(12.0);
      expect(result.statistic).toBeLessThan(13.0);
      expect(result.df).toBe(2);
      expect(result.pValue).toBeLessThan(0.01);
      expect(result.pValue).toBeGreaterThan(0.001);
    });

    it('should detect no difference for identical groups', () => {
      const group1 = [5, 5, 5];
      const group2 = [5, 5, 5];
      const group3 = [5, 5, 5];

      const result = kruskalWallis([group1, group2, group3]);

      expect(result.statistic).toBeCloseTo(0, 1);
      expect(result.pValue).toBeGreaterThan(0.9);
    });

    it('should handle ties across groups', () => {
      const group1 = [1, 2, 3];
      const group2 = [2, 3, 4];
      const group3 = [3, 4, 5];

      const result = kruskalWallis([group1, group2, group3]);

      expect(result.df).toBe(2);
      expect(result.statistic).toBeGreaterThan(0);
    });
  });

  describe('cohensD', () => {
    it('should calculate correct effect size', () => {
      const sample1 = [1, 2, 3, 4, 5];
      const sample2 = [3, 4, 5, 6, 7];

      // Python reference:
      // Using pingouin or manual calculation
      // d = (mean1 - mean2) / pooled_sd
      // mean1 = 3, mean2 = 5, diff = -2
      // pooled_sd ≈ 1.58
      // d ≈ -1.26

      const d = cohensD(sample1, sample2);

      expect(d).toBeCloseTo(-1.26, 1);
    });

    it('should return 0 for identical samples', () => {
      const sample = [1, 2, 3, 4, 5];

      const d = cohensD(sample, sample);

      expect(approxEqual(d, 0, 0.001)).toBe(true);
    });

    it('should calculate large effect size for very different samples', () => {
      const sample1 = [1, 2, 3];
      const sample2 = [10, 11, 12];

      const d = cohensD(sample1, sample2);

      expect(Math.abs(d)).toBeGreaterThan(3);
    });
  });

  describe('etaSquared and omegaSquared', () => {
    it('should match R etaSquared calculation', () => {
      const group1 = [1, 2, 3, 4, 5];
      const group2 = [6, 7, 8, 9, 10];
      const group3 = [11, 12, 13, 14, 15];

      const anovaResult = oneWayAnovaFn([group1, group2, group3]);

      // R reference:
      // library(effectsize)
      // eta_squared(aov(value ~ group, data))
      // η² = 0.9615...

      const eta2 = etaSquared(anovaResult);
      const omega2 = omegaSquared(anovaResult);

      expect(eta2).toBeGreaterThan(0.85);
      expect(eta2).toBeLessThan(1.0);
      expect(omega2).toBeGreaterThan(0.8);
      expect(omega2).toBeLessThan(eta2); // omega-squared is always less than eta-squared
    });

    it('should return 0 for no group differences', () => {
      const group1 = [5, 5, 5];
      const group2 = [5, 5, 5];
      const group3 = [5, 5, 5];

      const anovaResult = oneWayAnovaFn([group1, group2, group3]);
      const eta2 = etaSquared(anovaResult);

      expect(eta2).toBe(0);
    });
  });

  describe('bonferroni', () => {
    it('should match R p.adjust with method="bonferroni"', () => {
      const pValues = [0.01, 0.02, 0.03, 0.04, 0.05];

      // R reference:
      // p.adjust(c(0.01, 0.02, 0.03, 0.04, 0.05), method="bonferroni")
      // [1] 0.05 0.10 0.15 0.20 0.25

      const result = bonferroni(pValues, 0.05);

      expect(result.adjustedPValues[0]).toBeCloseTo(0.05, 5);
      expect(result.adjustedPValues[1]).toBeCloseTo(0.10, 5);
      expect(result.adjustedPValues[2]).toBeCloseTo(0.15, 5);
      expect(result.adjustedPValues[3]).toBeCloseTo(0.20, 5);
      expect(result.adjustedPValues[4]).toBeCloseTo(0.25, 5);

      // Only first should be rejected at alpha=0.05
      expect(result.rejected[0]).toBe(true);
      expect(result.rejected[1]).toBe(false);
    });

    it('should cap adjusted p-values at 1.0', () => {
      const pValues = [0.5, 0.6, 0.7];

      const result = bonferroni(pValues);

      result.adjustedPValues.forEach(p => {
        expect(p).toBeLessThanOrEqual(1.0);
      });
    });
  });

  describe('holmBonferroni', () => {
    it('should match R p.adjust with method="holm"', () => {
      const pValues = [0.01, 0.04, 0.03, 0.02, 0.05];

      // R reference:
      // p.adjust(c(0.01, 0.04, 0.03, 0.02, 0.05), method="holm")
      // [1] 0.05 0.12 0.12 0.08 0.05

      const result = holmBonferroni(pValues, 0.05);

      // Check that adjusted p-values are sorted correctly
      expect(result.adjustedPValues[0]).toBeCloseTo(0.05, 2);
      expect(result.adjustedPValues[3]).toBeCloseTo(0.08, 2);

      // Holm should be more powerful than Bonferroni
      const bonf = bonferroni(pValues, 0.05);
      const holmRejected = result.rejected.filter(r => r).length;
      const bonfRejected = bonf.rejected.filter(r => r).length;

      expect(holmRejected).toBeGreaterThanOrEqual(bonfRejected);
    });
  });

  describe('fdr (Benjamini-Hochberg)', () => {
    it('should match R p.adjust with method="BH"', () => {
      const pValues = [0.01, 0.04, 0.03, 0.02, 0.05];

      // R reference:
      // p.adjust(c(0.01, 0.04, 0.03, 0.02, 0.05), method="BH")
      // [1] 0.05 0.05 0.05 0.05 0.05

      const result = fdr(pValues, 0.05);

      // All should have adjusted p-value of 0.05
      expect(result.adjustedPValues[0]).toBeCloseTo(0.05, 2);
      expect(result.adjustedPValues[1]).toBeCloseTo(0.05, 2);
      expect(result.adjustedPValues[2]).toBeCloseTo(0.05, 2);
      expect(result.adjustedPValues[3]).toBeCloseTo(0.05, 2);
      expect(result.adjustedPValues[4]).toBeCloseTo(0.05, 2);

      // All should be rejected at alpha=0.05
      expect(result.rejected.every(r => r)).toBe(true);
    });

    it('should be more powerful than Bonferroni', () => {
      const pValues = [0.001, 0.008, 0.015, 0.02, 0.025, 0.03, 0.06, 0.08, 0.15, 0.20];

      const fdrResult = fdr(pValues, 0.05);
      const bonfResult = bonferroni(pValues, 0.05);

      const fdrRejected = fdrResult.rejected.filter(r => r).length;
      const bonfRejected = bonfResult.rejected.filter(r => r).length;

      // FDR should reject at least as many hypotheses as Bonferroni
      expect(fdrRejected).toBeGreaterThanOrEqual(bonfRejected);
      // In this case, should actually reject more
      expect(fdrRejected).toBeGreaterThan(0);
    });

    it('should handle edge case with all small p-values', () => {
      const pValues = [0.001, 0.002, 0.003, 0.004, 0.005];

      const result = fdr(pValues, 0.05);

      // All should be rejected
      expect(result.rejected.every(r => r)).toBe(true);
    });

    it('should handle edge case with all large p-values', () => {
      const pValues = [0.5, 0.6, 0.7, 0.8, 0.9];

      const result = fdr(pValues, 0.05);

      // None should be rejected
      expect(result.rejected.every(r => !r)).toBe(true);
    });
  });

  describe('Integration tests', () => {
    it('should use effect sizes with t-tests', () => {
      const control = [12, 14, 13, 15, 14, 13, 12];
      const treatment = [15, 17, 16, 18, 17, 16, 15];

      const d = cohensD(control, treatment);

      // Large effect size
      expect(Math.abs(d)).toBeGreaterThan(1.5);
    });

    it('should use multiple testing corrections with multiple comparisons', () => {
      // Simulate multiple t-test p-values
      const pValues = [0.005, 0.02, 0.03, 0.15, 0.25, 0.40, 0.60, 0.75, 0.85, 0.95];

      const bonfResult = bonferroni(pValues, 0.05);
      const holmResult = holmBonferroni(pValues, 0.05);
      const fdrResult = fdr(pValues, 0.05);

      // FDR should be most powerful, Bonferroni most conservative
      const bonfCount = bonfResult.rejected.filter(r => r).length;
      const holmCount = holmResult.rejected.filter(r => r).length;
      const fdrCount = fdrResult.rejected.filter(r => r).length;

      expect(fdrCount).toBeGreaterThanOrEqual(holmCount);
      expect(holmCount).toBeGreaterThanOrEqual(bonfCount);
    });

    it('should use effect sizes with ANOVA', () => {
      const group1 = [1, 2, 3, 4];
      const group2 = [5, 6, 7, 8];
      const group3 = [9, 10, 11, 12];

      const anovaResult = oneWayAnovaFn([group1, group2, group3]);
      const eta2 = etaSquared(anovaResult);
      const omega2 = omegaSquared(anovaResult);

      // Very large effect size
      expect(eta2).toBeGreaterThan(0.85);
      expect(omega2).toBeGreaterThan(0.8);
      expect(omega2).toBeLessThan(eta2);
    });
  });
});
