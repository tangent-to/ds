// ---
// title: Statistical analysis
// id: ds-statistics
// ---

// %% [markdown]
/*
# Statistical analysis

Statistics helps us make sense of data by summarizing patterns, testing hypotheses, and modeling relationships. They are mostly used to investigate the process that have generated the data, what can be associated with a *detective* mode. This notebook covers three essential statistical approaches with `tangent/ds`.

- Descriptive statistics summarize and describe your data.
- Hypothesis testing creates probabilistic statistics, in *frequential* mode.
- Linear models establishes relationships between variables.

While ordination methods (PCA, LDA, RDA) help us visualize patterns, statistics helps us

- to quantify uncertainty,
- to test probabilities on patterns,
- to model relationships between variables

If the aim is to *predict* outcomes, you might prefer machine learning.
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
## Descriptive statistics

Descriptive statistics summarize data with a few key numbers, usually central tendency like mean or median, and the spread like standard deviatio, quantiles and ranges.
*/

// %% [javascript]

const beakLength = penguinsData.map(d => d["Beak Length (mm)"])

// %% [javascript]

ds.core.math.mean(beakLength, { naOmit: true })

// %% [javascript]

ds.core.math.stddev(beakLength, { naOmit: true })

// %% [javascript]

ds.core.math.quantile(beakLength, [0.25, 0.5, 0.75], { naOmit: true })

// %% [markdown]
/*
### Grouped statistics with Arquero

The [Arquero package](https://observablehq.com/@uwdata/introducing-arquero) provides a declarative approach for computing statistics by group (similar to dplyr in R or pandas in Python). This is especially useful when you want to compare statistics across categories.
*/

// %% [javascript]

// Adapted for the browser: arquero is loaded from a CDN instead of a dynamic import.
import * as aq from 'https://esm.sh/arquero';

// %% [javascript]

const penguinsTable = aq.from(penguinsData);

// %% [javascript]

void penguinsTable
  .groupby('Species')
  .rollup({
    min:  d => op.min(d["Beak Length (mm)"]), // functional form of op.min('sun')
    max:  d => op.max(d["Beak Length (mm)"]),
    mean:  d => op.average(d["Beak Length (mm)"]),
    median:  d => op.median(d["Beak Length (mm)"]),
    sd:  d => op.stdev(d["Beak Length (mm)"]),
  })
  .print()

// %% [markdown]
/*
## Hypothesis testing

Hypothesis testing answers a statistical question. Ready for some jargon? The *statistical model* is the tool used for the test (t-test, anova, glm, etc.), and how the analyst designed the model. The *null hypothesis* (H₀) generally assumes the statu quo, that the action like a treatment has no effect in the population. The *alternative hypothesis* (H₁) assumes the complementary, that the action has sn effect in the population. Both H₀ and H₁ are valid only in the frame of the statistical model. A *test statistic* computes a number measuring the effect.

The *p-value* is an important test statistic. It's the probability that the sample was taken from a population under H₀.

- **P-value ≠ probability that H₀ is true**.
- P-values depend a lot on sample size. Since larger data sets generate smaller p-values, significance (when p-value < 0.05) doesn't mean variable importance.
- Always report effect sizes and confidence intervals, never just p-values - which are accessory.
- The p-value threshold of 0.05 is arbitrary.

Our Question: Do penguin species differ in body bass? Let's test if Adelie, Chinstrap, and Gentoo penguins have different average body masses by first extracting them.
*/

// %% [javascript]

const tested_variable = "Body Mass (g)";
const adelie_var = penguinsData
  .filter((d) => d.Species == "Adelie")
  .map((d) => d[tested_variable]);
const chinstrap_var = penguinsData
  .filter((d) => d.Species == "Chinstrap")
  .map((d) => d[tested_variable]);
const gentoo_var = penguinsData
  .filter((d) => d.Species == "Gentoo")
  .map((d) => d[tested_variable]);

// %% [markdown]
/*
## T-test

A t-test compares means of two groups. The null hypothesis is that the two groups have the same mean. Let's compare Adelie vs Gentoo body mass to test if they are different (a two-sided test). If the test was about sample 1 being less than sample 2, the alternative hypothesis would be `less`.
*/

// %% [javascript]

const ttest = ds.stats.hypothesis.twoSampleTTest(adelie_var, gentoo_var, { alternative: 'two-sided' })
console.table(ttest);

// %% [markdown]
/*
Interpreting the T-Test results:

- **statistic = -23.5**: Large negative value → Adelie mean < Gentoo mean
- **pValue ≈ 0.002**: Very small → Evidence against H₀
- **mean1 = 3606g, mean2 = 5092g**: Gentoo penguins are ~1359g heavier!
- **Conclusion**: Adelie and Gentoo penguins have significantly different body masses
*/

// %% [markdown]
/*
## ANOVA

Although its name refers to variance, ANOVA (analysis of variance) tests H₀, assuming samples from multiple groups are drawn from populations sharing the same mean. If the groups differ, the variance *between* groups should be large compared to the variance *within* groups. This comparison is where the variance aspect of ANOVA comes into play. When we are comparing *one* feature (e.g., body mass) across groups (e.g., different species), we perform a *one*-way ANOVA. ANOVA can be extended to two-way, three-way, and even higher (e.g., seven-hundred ninety-nine-way ANOVA) depending on how many factors are involved. Each additional factor complicates the analysis and adds more interaction terms to consider. This highlights the fact that ANOVA can be viewed as a case of linear regression when the target variable is categorical. In essence, it assesses how well categorical predictors help explain the variability in the response variable.
*/

// %% [javascript]

const anova = ds.stats.hypothesis.oneWayAnova([adelie_var, chinstrap_var, gentoo_var]);
console.table(anova);

// %% [markdown]
/*
Interpreting ANOVA results:

- **statistic (F) ≈ 342**, very large → strong between-group differences
- **pValue ≈ 0**, essentially zero → overwhelming evidence of differences
- **MSbetween vs MSwithin**, between-group variance is ~218× larger than within-group
- **Conclusion**: at least one species differs significantly from the others. ANOVA tells us groups differ, but not *which* groups. For that, we'd need post-hoc tests (e.g., Tukey HSD).
*/

// %% [markdown]
/*
## Tukey

The ANOVA indicated significant differences exist, and Tukey's test identifies which specific pairs differ by doing a collection of pairwise t-tests.
*/

// %% [javascript]

// reuses the anova computed in the ANOVA cell above
const tukey = ds.stats.hypothesis.tukeyHSD({Adelie: adelie_var, Chintrap: chinstrap_var, Gentoo: gentoo_var}, { alpha: 0.05, anovaResult: anova })
tukey;

// %% [markdown]
/*
The Tukey test results returns important information on group means, showing that gentoos are much larger. Remember that the test is not about the importance of the difference, but the importance of the variation around the differences. In the context of bird species, although, it would be surprising to find so much variance in body mass (this prior knowledge isn't included in the statistical model, to do so we would need bayesian statistics, which `tangent/ds` isn't equipped to do yet). The pairwise test between adelie and chinstrap shows to be small with a confidence interval crossing the zero widelybetween -256 and +202 g, with a mean of -26.92 g. The test shows that the mass of both populations are likely drawn from populations with similar means of body masses. However, the difference between gentoo and adelie, and between gentoo and chinstrap are far from crossing zero. Data shows that gentoo penguins are substantially heavier than both adelie and chinstrap penguins, while adelie and chinstrap penguins have similar body masses.
*/

// %% [markdown]
/*
## Linear models

Linear models describe relationships between variables as $y = β₀ + β₁x₁ + β₂x₂ + ... + ε$. A Generalized Linear Model (GLM) extends classical linear regressions with a continuous `y` to different types of response variables.

| Type | Family | Link | Use For |
|------|--------|------|----------|
| Linear Regression | gaussian | identity | Continuous Y |
| Logistic Regression | binomial | logit | Binary Y (0/1) |
| Poisson Regression | poisson | log | Count Y (0,1,2,...) |

In `tangent/ds`, all linear models use the `GLM` class - you just specify the family and link function.
*/

// %% [markdown]
/*
### Simple linear regression

Let's model the relationship between beak length and body mass, asking does beak length predict body mass under $Body~mass = β₀ + β₁ × Beak~length + ε$

We use `family: 'gaussian'` (normal distribution) and `link: 'identity'` (direct linear relationship).
*/

// %% [javascript]

const simple_lm = new ds.stats.GLM({family: 'gaussian', link: 'identity'});
simple_lm.fit({
  X: "Beak Length (mm)",
  y: 'Body Mass (g)',
  data: penguinsData,
});
console.log(simple_lm.summary({ alpha: 0.05 })); // use console.log() for a prettier output

// %% [markdown]
/*
The `Estimate` column are the values of $β_p$. The `(Intercept)` of 388.8* is the value of $β_0$, the body mass when beak length = 0. Having a beak length of 0 in an extrapolation of the domain, and the intercept is not meaningful here. The `Estimate` of beak length is the slope, of `87.4`.  This means that for each 1 mm increase in beak length, body mass increases by ~87g. The `95% CI = [74.9, 100.0]` is the confidence range of the effect of beak length. We are 95% confident that the slope of the population is in this range, and the CI doesn't cross 0. The slope estimate depends on the scale at which variables are expressed, so they can't be compared one to another. Z-value are standardized, and can be use to compare the importance of variables in a GLM - here we just have two estimates. The pseudo R² of 0.354 tells us that beak length explains 35.4% of variance in body mass. Interpreting if it's high or low really depends on the use case. AIC/BIC are mostly use to compare models one to another. Not so useful if the definition of the model is fixed, but useful to assess the features we might want to drop. The model tells us that penguins with longer beaks tend to be heavier. But R² = 0.354 is far from 1 and means beak length alone isn't the whole story.
*/

// %% [markdown]
/*
#### About P-values

By default, the summary doesn't show p-values. This is intentional, because [p-values are notoriously misused](https://en.wikipedia.org/wiki/Misuse_of_p-values). Focus on:
1. Effect sizes (coefficients)
2. Confidence intervals
3. R² (variance explained)

But if you need them, they're computed silently:
*/

// %% [javascript]

simple_lm.pvalues();

// %% [markdown]
/*
#### Making predictions

Once we have a fitted model, we can predict new values. Let's predict body mass for a penguin with a 40 mm beak.
*/

// %% [javascript]

simple_lm.predict({
  X: "Beak Length (mm)",
  data: [{"Beak Length (mm)": 40}]
});

// %% [markdown]
/*
Predictions are useful to plot the GLM.
*/

// %% [javascript]

// Get min and max, automatically omitting NaN/null/undefined
const xMin = ds.core.math.min(penguinsData.map(d => d["Beak Length (mm)"]), { naOmit: true });
const xMax = ds.core.math.max(penguinsData.map(d => d["Beak Length (mm)"]), { naOmit: true });

// Generate sequence for prediction
const xValues = ds.core.math.range(xMin, xMax, 0.5).map(x => ({
  "Beak Length (mm)": x
}));

// predict
const predictions = lm.predict({
  X: "Beak Length (mm)",
  data: xValues
});

// Combine x values with predictions
const lineData = xValues.map((d, i) => ({
  x: d["Beak Length (mm)"],
  y: predictions[i]
}));

// Create the plot
const plot = Plot.plot({
  marks: [
    // Scatter plot of actual data
    Plot.dot(penguinsData, {
      x: "Beak Length (mm)",
      y: "Body Mass (g)",
      fill: "steelblue",
      opacity: 0.6
    }),
    // Regression line
    Plot.line(lineData, {
      x: "x",
      y: "y",
      stroke: "red",
      strokeWidth: 2
    })
  ],
  grid: true,
  x: { label: "Beak Length (mm)" },
  y: { label: "Body Mass (g)" }
});
plot;

// %% [markdown]
/*
### Categorical Target

When the target variable is categorical (e.g., species), we use **logistic regression** instead of linear regression. Logistic models differ from gaussian models in several ways.

- **Family**: `binomial` instead of `gaussian`
- **Link function**: `logit` transforms probabilities to log-odds: `logit(p) = log(p / (1-p))`
- **Output**: Probabilities between 0 and 1 (after inverse-logit)
- **Interpretation**: Coefficients represent change in log-odds, not raw values

To perform categorical linear models, we use a multiclass strategy. For 3 or more classes (like Adelie, Chinstrap, Gentoo), we use a one-vs-rest (OVR) model:

- Fit 3 separate binary models: "Is Adelie?", "Is Chinstrap?", "Is Gentoo?".
- Each model predicts probability of being that class vs all others.
- Final prediction: class with highest probability.

Categorical variables must be converted to numbers (one-hot encoding).
*/

// %% [javascript]

const encoder = new ds.core.table.OneHotEncoder();
const encoded = encoder.fitTransform({
    data: penguinsData,
    columns: ["Species", "Sex"]
});
const penguinsWithOneHot = penguinsData.map((row, i) => ({
    ...row,
    ...encoded[i]
}));
console.table(penguinsWithOneHot.slice(0, 6));

// %% [javascript]

const model = new ds.stats.GLM({
  family: "binomial",
  multiclass: "ovr" // use ovr for a collection of binomial regressions with one group versus others, and multinomial to predict class assignment probabilities
});

model.fit({
  X: ["Beak Length (mm)", "Beak Depth (mm)", "Flipper Length (mm)", "Body Mass (g)", "Sex_FEMALE"], // use only one Sex_ variable since categories has d-1 information, since male is non-female.
  y: "Species", // don't use multiple target with ovr
  data: penguinsWithOneHot // it will get the Species_ columns
});
model

// %% [markdown]
/*
### Mixed-effects models, aka hierarchical models

Sometimes data has nested structure or grouping which are defined by the research methodology, not but the property of the studied object. The difference between a fixed and a random effect is that the sum of all random effect is forced to be null. A model structure to test for yield against barley's variety and year with site as random effect could be expressed as

$yield = β₀ + β₁·variety + u_{site} + ε$

where:
- $β₀, β₁$ = fixed intercept and slopes (same for all sites)
- $u_{site}$ = random intercept for each site (allows baseline yield to vary by site)
- $ε$ = residual error
*/

// %% [javascript]

const response = await fetch('https://cdn.jsdelivr.net/npm/vega-datasets@2/data/barley.json');
const barleyData = await response.json();
console.table(barleyData.slice(0, 6));

// %% [markdown]
/*
The variety being a category, it must be encoded.
*/

// %% [javascript]

const barleyEncoder = new ds.core.table.OneHotEncoder();
const barleyVarietyEncoded = barleyEncoder.fitTransform({
    data: barleyData,
    columns: ["variety"]
});
const barleyDataEncoded = barleyData.map((row, i) => ({
    ...row,
    ...barleyVarietyEncoded[i]
}));
barleyDataEncoded[50];

// %% [markdown]
/*
The features of the models are the different varieties, except the extra information provided by one of them (if all otehrs are zeros, than we know it's the removed one, no need to argue on that). We remove `variety_Manchuria`, used as the reference, the one used for the intercept.
*/

// %% [javascript]

const featuresArray = [
  "variety_Glabron",
  "variety_Svansota",
  "variety_Velvet",
  "variety_Trebi",
  "variety_No. 457",
  "variety_No. 462",
  "variety_Peatland",
  "variety_No. 475",
  "variety_Wisconsin No. 38"
]

// %% [markdown]
/*
Mixed effects models use the same GLM API we used yet, but adding a `randomEffects` argument. Here we consider that the methodological effect of sites is on the intercept, supposing they have a simple additive effect on yield.
*/

// %% [javascript]

const sites = barleyDataEncoded.map((d) => d.site);
const years = barleyDataEncoded.map((d) => d.year);

const lmmEstimator = new ds.stats.GLM({
  family: 'gaussian',
  randomEffects: {
    // random intercept for each site
    intercept: sites,
    // each site is allowed to have its own slope for the predictor `year`
    //slopes: {
    //  year: {
    //    groups: sites,
    //    values: years
    //  }
    //}
  }
});

lmmEstimator.fit({
  data: barleyDataEncoded,
  X: featuresArray,   // include "year" along with your other fixed effects
  y: 'yield'
});

console.log(lmmEstimator.summary());

// %% [markdown]
/*
Interpreting mixed-effects model results needs time and concentration. The fixed effects are the effects of varieties (compared to Manchuria baseline) on yield. The manchura variety has a sample mean yield of 31.4 bushels per acre, and we see the relative effect of other varieties compared to this intercept, e.g. trebi = +7.94 more than manchuri. These are population-level effects that hold across all sites. Random effects are site-to-site variability, with standard deviation of 7.43. Sites differ by ±7.4 bushels/acre on average among  sites. The random effect accounts for the fact that some sites are naturally more productive than others (soil quality, climate, etc.). Without accounting for this, our variety estimates would be biased.
*/
