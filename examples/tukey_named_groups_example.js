// ---
// title: Tukey HSD with named groups
// id: ds-tukey-named-groups
// ---

// %% [markdown]
/*
# Tukey HSD with named groups

A one-way ANOVA answers a single yes/no question: is *any* group mean
different from the others? When the answer is yes, it does not say *which*
groups differ. Tukey's Honestly Significant Difference (HSD) test fills that
gap by comparing every pair of groups while controlling the family-wise error
rate across all the comparisons.

`ds.stats.hypothesis.tukeyHSD` accepts groups either as a bare array (labelled
`Group 0`, `Group 1`, ...) or as an object mapping names to samples, which
carries readable labels through to the results. We show both on penguin body
masses.
*/

// %% [javascript]

import ds from 'https://esm.sh/@tangent.to/ds';

// Body mass (grams) for three penguin species.
const adelie = [3750, 3800, 3250, 3450, 3650, 3625, 4675, 3475, 4250, 3300];
const chinstrap = [3700, 3900, 3800, 3400, 3725, 3600, 3750];
const gentoo = [4750, 5700, 5400, 4800, 5200, 5400, 5650, 5700, 5900, 6300];

// The omnibus ANOVA first: is there any difference in mean mass at all?
const anova = ds.stats.hypothesis.oneWayAnova([adelie, chinstrap, gentoo]);

({
  F: anova.statistic,     // ~59.7
  pValue: anova.pValue,   // ~4.8e-10, so yes
  dfBetween: anova.dfBetween,
  dfWithin: anova.dfWithin,
});

// %% [markdown]
/*
## Array input

Passing an array of samples runs every pairwise comparison. We hand the ANOVA
result in via `anovaResult` so Tukey reuses its within-group mean square. Each
comparison reports the mean difference, a confidence interval, a p-value
adjusted for multiplicity, and a `significant` flag. With array input the
groups are named positionally.
*/

// %% [javascript]

const tukeyArray = ds.stats.hypothesis.tukeyHSD(
  [adelie, chinstrap, gentoo],
  { alpha: 0.05, anovaResult: anova },
);

({
  groupNames: tukeyArray.groupNames, // [Group 0, Group 1, Group 2]
  comparisons: tukeyArray.comparisons.map((c) => ({
    pair: c.groupLabels.join(' vs '),
    pValue: c.pValue,
    significant: c.significant,
  })),
});

// %% [markdown]
/*
## Named groups

Passing an object instead keeps the species names attached to every result,
which is what you want in a report. The statistics are identical to the array
call; only the labels change. Adelie and Chinstrap are statistically
indistinguishable in mass (p = 1), while Gentoo is heavier than both by a wide,
highly significant margin.
*/

// %% [javascript]

const tukeyNamed = ds.stats.hypothesis.tukeyHSD(
  { Adelie: adelie, Chinstrap: chinstrap, Gentoo: gentoo },
  { alpha: 0.05, anovaResult: anova },
);

({
  groupNames: tukeyNamed.groupNames, // [Adelie, Chinstrap, Gentoo]
  comparisons: tukeyNamed.comparisons.map((c) => ({
    pair: c.groupLabels.join(' vs '),
    pValue: c.pValue,
    significant: c.significant,
  })),
});

// %% [markdown]
/*
## The full detail for one comparison

Each comparison object carries everything you need to report the result: the
group labels, the estimated mean difference, its 95% confidence interval, the
studentised-range q statistic, the adjusted p-value, and the significance
decision. Here is the Chinstrap-vs-Gentoo pair in full.
*/

// %% [javascript]

const chinstrapVsGentoo = tukeyNamed.comparisons.find(
  (c) => c.groupLabels.includes('Chinstrap') && c.groupLabels.includes('Gentoo'),
);

({
  pair: chinstrapVsGentoo.groupLabels.join(' vs '),
  meanDifference: chinstrapVsGentoo.diff,
  ci95: [chinstrapVsGentoo.lowerCI, chinstrapVsGentoo.upperCI],
  qStatistic: chinstrapVsGentoo.qStatistic,
  pValue: chinstrapVsGentoo.pValue,
  significant: chinstrapVsGentoo.significant,
});
