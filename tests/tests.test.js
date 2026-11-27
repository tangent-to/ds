import { describe, it, expect } from 'vitest';
import { OneSampleTTest, TwoSampleTTest, ChiSquareTest, OneWayAnova, TukeyHSD } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('statistical tests', () => {
  describe('oneSampleTTest', () => {
    it('should perform one-sample t-test', () => {
      const sample = [5, 6, 7, 8, 9];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 5 });
      const result = test.summary();
      
      expect(result.mean).toBe(7);
      expect(result.df).toBe(4);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
    });

    it('should test against different hypothesized mean', () => {
      const sample = [10, 11, 12, 13, 14];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 12 });
      const result = test.summary();
      
      expect(result.mean).toBe(12);
      expect(approxEqual(result.pValue, 1, 0.5)).toBe(true); // should be close to 1
    });

    it('should support one-sided tests', () => {
      const sample = [10, 11, 12, 13, 14];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 8, alternative: 'greater' });
      const resultLess = test.summary();
      expect(resultLess.pValue).toBeLessThan(0.5);
    });
  });

  describe('twoSampleTTest', () => {
    it('should perform two-sample t-test', () => {
      const sample1 = [1, 2, 3, 4, 5];
      const sample2 = [3, 4, 5, 6, 7];
      const test = new TwoSampleTTest();
      test.fit(sample1, sample2);
      const result = test.summary();
      
      expect(result.mean1).toBe(3);
      expect(result.mean2).toBe(5);
      expect(result.df).toBe(8);
      expect(result.statistic).toBeLessThan(0); // mean1 < mean2
    });

    it('should detect no difference when samples are similar', () => {
      const sample1 = [5, 6, 7, 8, 9];
      const sample2 = [5, 6, 7, 8, 9];
      const test = new TwoSampleTTest();
      test.fit(sample1, sample2);
      const result = test.summary();
      
      expect(approxEqual(result.statistic, 0, 0.001)).toBe(true);
    });
  });

  describe('chiSquareTest', () => {
    it('should perform chi-square goodness of fit test', () => {
      const observed = [10, 20, 30];
      const expected = [15, 20, 25];
      const test = new ChiSquareTest();
      test.fit(observed, expected);
      const result = test.summary();
      
      expect(result.df).toBe(2);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
    });

    it('should return perfect fit for identical distributions', () => {
      const observed = [10, 20, 30];
      const expected = [10, 20, 30];
      const test = new ChiSquareTest();
      test.fit(observed, expected);
      const result = test.summary();
      
      expect(result.statistic).toBe(0);
    });
  });

  describe('oneWayAnova', () => {
    it('should perform one-way ANOVA', () => {
      const group1 = [1, 2, 3];
      const group2 = [4, 5, 6];
      const group3 = [7, 8, 9];
      const test = new OneWayAnova();
      test.fit([group1, group2, group3]);
      const result = test.summary();
      
      expect(result.dfBetween).toBe(2);
      expect(result.dfWithin).toBe(6);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.MSbetween).toBeGreaterThan(result.MSwithin);
    });

    it('should detect no difference for identical groups', () => {
      const group1 = [5, 5, 5];
      const group2 = [5, 5, 5];
      const group3 = [5, 5, 5];
      const test = new OneWayAnova();
      test.fit([group1, group2, group3]);
      const result = test.summary();

      expect(result.MSbetween).toBe(0);
    });
  });

  describe('tukeyHSD', () => {
    it.each([
      {
        name: 'balanced groups',
        groups: [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
        alpha: 0.05,
        expected: [
          [0.5907706083, -3.6678636521, 1.6678636521],
          [0.1545799684, -4.6678636521, 0.6678636521],
          [0.5907706083, -3.6678636521, 1.6678636521],
        ],
      },
      {
        name: 'unequal group sizes and 99% confidence',
        groups: [[2, 4], [3, 5, 9], [6, 7, 8, 10, 12], [2, 3, 4, 4]],
        alpha: 0.01,
        expected: [
          [0.5531447822, -10.7062727103, 5.3729393769],
          [0.0468672255, -12.9684206505, 1.7684206505],
          [0.9990869458, -7.8770399765, 7.3770399765],
          [0.3029950766, -9.3650181682, 3.4983515015],
          [0.4909351747, -4.3097503391, 9.1430836724],
          [0.0182542649, -0.5578797619, 11.2578797619],
        ],
      },
      {
        name: 'two groups',
        groups: [[1, 3, 9], [2, 4, 6, 10]],
        alpha: 0.10,
        expected: [[0.6993154670, -6.9114200119, 4.5780866785]],
      },
    ])('matches R TukeyHSD for $name', ({ groups, alpha, expected }) => {
      // R's TukeyHSD(aov(y ~ group), conf.level = 1 - alpha) reports j − i.
      // The intervals below use this API's i − j convention and pair order.
      const { comparisons } = new TukeyHSD({ alpha }).fit(groups).summary();
      expect(comparisons).toHaveLength(expected.length);
      comparisons.forEach(({ pValue, lowerCI, upperCI, significant }, i) => {
        const [p, lower, upper] = expected[i];
        expect(pValue).toBeCloseTo(p, 7);
        expect(lowerCI).toBeCloseTo(lower, 7);
        expect(upperCI).toBeCloseTo(upper, 7);
        expect(significant).toBe(p < alpha);
      });
    });

    it('handles zero pooled variance and rejects undefined pooled variance', () => {
      const { comparisons } = TukeyHSD.compute([[1, 1], [1, 1], [3, 3]]);
      expect(comparisons[0]).toMatchObject({ pValue: 1, qStatistic: 0, lowerCI: 0, upperCI: 0 });
      expect(comparisons[1]).toMatchObject({ pValue: 0, qStatistic: Infinity, lowerCI: -2, upperCI: -2 });
      expect(() => TukeyHSD.compute([[1], [3]])).toThrow('positive residual degrees of freedom');
    });

    it('resolves the upper tail with one residual degree of freedom', () => {
      // scipy.stats.studentized_range.sf(30, 3, 1) and isf(0.05, 3, 1).
      const { comparisons, dfWithin } = TukeyHSD.compute([[-1, 1], [0], [30 * Math.SQRT2]]);
      const { pValue, qStatistic, diff, upperCI, significant } = comparisons[2];
      expect(dfWithin).toBe(1);
      expect(qStatistic).toBeCloseTo(30);
      expect(pValue).toBeCloseTo(0.0449700367, 8);
      expect((upperCI - diff) / Math.SQRT2).toBeCloseTo(26.9755298695, 6);
      expect(significant).toBe(true);
    });

    it('should perform Tukey HSD post-hoc test', () => {
      const group1 = [1, 2, 3];
      const group2 = [4, 5, 6];
      const group3 = [7, 8, 9];
      const test = new TukeyHSD();
      test.fit([group1, group2, group3]);
      const result = test.summary();

      expect(result.numGroups).toBe(3);
      expect(result.comparisons.length).toBe(3); // 3 choose 2 = 3 comparisons
      expect(result.groupMeans).toEqual([2, 5, 8]);
      expect(result.alpha).toBe(0.05);

      // Check structure of comparisons
      const comp = result.comparisons[0];
      expect(comp).toHaveProperty('groups');
      expect(comp).toHaveProperty('diff');
      expect(comp).toHaveProperty('lowerCI');
      expect(comp).toHaveProperty('upperCI');
      expect(comp).toHaveProperty('pValue');
      expect(comp).toHaveProperty('significant');
      expect(comp).toHaveProperty('qStatistic');
    });

    it('should detect significant differences between distinct groups', () => {
      const group1 = [1, 2, 3, 4, 5];
      const group2 = [10, 11, 12, 13, 14];
      const group3 = [20, 21, 22, 23, 24];
      const test = new TukeyHSD();
      test.fit([group1, group2, group3]);
      const result = test.summary();

      // All comparisons should be significant
      result.comparisons.forEach(comp => {
        expect(comp.pValue).toBeLessThan(0.05);
        expect(comp.significant).toBe(true);
      });
    });

    it('should not detect differences for similar groups', () => {
      const group1 = [5, 6, 7];
      const group2 = [5, 6, 7];
      const group3 = [5, 6, 7];
      const test = new TukeyHSD();
      test.fit([group1, group2, group3]);
      const result = test.summary();

      // All differences should be zero
      result.comparisons.forEach(comp => {
        expect(approxEqual(comp.diff, 0, 0.001)).toBe(true);
      });
    });

    it('should support custom alpha level', () => {
      const group1 = [1, 2, 3];
      const group2 = [4, 5, 6];
      const test = new TukeyHSD({ alpha: 0.01 });
      test.fit([group1, group2]);
      const result = test.summary();

      expect(result.alpha).toBe(0.01);
    });

    it('should work with precomputed ANOVA result', () => {
      const groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const anova = new OneWayAnova();
      anova.fit(groups);
      const anovaResult = anova.summary();

      const test = new TukeyHSD();
      test.fit(groups, { anovaResult });
      const result = test.summary();

      expect(result.MSwithin).toBe(anovaResult.MSwithin);
      expect(result.dfWithin).toBe(anovaResult.dfWithin);
    });

    it('should throw error with less than 2 groups', () => {
      const test = new TukeyHSD();
      expect(() => test.fit([[1, 2, 3]])).toThrow('Need at least 2 groups');
    });

    it('should produce confidence intervals that contain zero for non-significant differences', () => {
      const group1 = [5, 6, 7, 8, 9];
      const group2 = [5.5, 6.5, 7.5, 8.5, 9.5];
      const test = new TukeyHSD();
      test.fit([group1, group2]);
      const result = test.summary();

      const comp = result.comparisons[0];
      // For small differences, CI should likely contain zero
      if (!comp.significant) {
        expect(comp.lowerCI).toBeLessThanOrEqual(0);
        expect(comp.upperCI).toBeGreaterThanOrEqual(0);
      }
    });

    it('should label groups correctly', () => {
      const group1 = [1, 2, 3];
      const group2 = [4, 5, 6];
      const group3 = [7, 8, 9];
      const test = new TukeyHSD();
      test.fit([group1, group2, group3]);
      const result = test.summary();

      // Check that group labels exist
      result.comparisons.forEach(comp => {
        expect(comp.groupLabels).toHaveLength(2);
        expect(comp.groupLabels[0]).toContain('Group');
        expect(comp.groupLabels[1]).toContain('Group');
      });
    });
  });
});
