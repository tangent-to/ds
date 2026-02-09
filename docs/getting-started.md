---
layout: default
title: Getting Started
nav_order: 2
description: "Quick start guide for tangent/ds - install, import, and run your first analysis"
permalink: /getting-started
---

# Getting started
{: .no_toc }

Get up and running with tangent/ds in minutes.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## No install

The easiest way to use tangent/ds is to head directly to Observablehq.com, then import the library in a JavaScript.

```javascript
ds = await import("https://esm.sh/@tangent.to/ds")
```

If you need some example to kick start your analysis, check out the data science collection [Data science with tangent/ds](https://observablehq.com/collection/@essi/data-science-with-tangent).

## Installation

If you prefer to work locally, you can install tangent/ds with deno, npm, or downloading it in the browser on refresh. 

### Deno

Deno supports Jupyter, the most widely used notebook interface of the multiverse. After installing Deno, write `deno jupyter --install` in the terminal to install Jupyter, `deno add @tangent/ds` to install tangent/ds, then launch your app with `jupyter`.

Once Jupyter is installed, you can also run the notebook in [Zed](https://zed.dev/). Create workspace settings (e.g., /.zed/settings.json) to prefer the Deno kernel and enable Deno LSP:

```
{
  "jupyter": { "kernel_selections": { "typescript": "deno", "javascript": "deno" } },
  "lsp": { "deno": { "settings": { "deno": { "enable": true } } } }
}
```

In Zed's deno notebooks, cells are separated by `// %%` tags as

```
// %%
import * as aq from "https://esm.sh/arquero@5.3.0";

// %%
const t = aq.table([{a:1},{a:2}]);
console.log(t.toString());
```

### NPM

```bash
npm install @tangent/ds
```

### CDN (Browser)

```html
<script type="module">
  import * as ds from 'https://cdn.jsdelivr.net/npm/@tangent/ds/+esm';
</script>
```

---

## First analysis

Let's run a simple t-test to compare two groups.

### 1. Import the library

```javascript
import * as ds from '@tangent/ds';
```

### 2. Prepare your data

```javascript
const penguinsResponse = await fetch(
  'https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json',
);
const penguinsData = await penguinsResponse.json();
console.table(penguinsData.slice(0, 5));
```

### 3. Run the analysis

```javascript
const tested_variable = "Body Mass (g)";

const adelie_var = penguinsData
  .filter((d) => d.Species == "Adelie")
  .map((d) => d[tested_variable]);

const chinstrap_var = penguinsData
  .filter((d) => d.Species == "Chinstrap")
  .map((d) => d[tested_variable]);

const ttest = ds.stats.hypothesis.twoSampleTTest(adelie_var, chinstrap_var);

console.log(ttest);
```

```
{
  statistic: -18.42963067630639,
  pValue: 0.0020000000000000018,
  df: 274,
  mean1: 3676.315789473684,
  mean2: 5035.080645161291,
  pooledSE: 73.72718854504609,
  alternative: "two-sided"
}
```

---

## Core concepts

### Declarative API

DS uses a declarative approach where you describe your data and analysis:

```javascript
const penguinsFeatureFields = [
  'Beak Length (mm)',
  'Beak Depth (mm)',
  'Flipper Length (mm)',
  'Body Mass (g)',
];

const pcaData = penguinsData.map(d =>
  penguinsFeatureFields.reduce((row, field) => {
    row[field] = d[field];
    return row;
  }, {})
);

const pca = new ds.mva.PCA({
  center: true,
  scale: true,
  scaling: 2, // correlation biplot
  omit_missing: true
});

pca.fit({data: pcaData});
```

### Fit-Transform Pattern

Many methods follow the **fit-transform** pattern from scikit-learn:

```javascript
const penguinsSplit = ds.ml.validation.trainTestSplit(
  { data: penguinsData, X: penguinsFeatureFields, y: 'Species' },
  { ratio: 0.7, shuffle: true, seed: 42 }
);

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

```

### Integration with Observable Plot

tagent/ds works seamlessly with Observable Plot for visualization:

```javascript
import * as Plot from '@observablehq/plot';

ds.plot.ordiplot(pcaEstimator.model, {
  colorBy: penguinsData.map(d => d.Species),
  showLoadings: true,
}).show(Plot);
```

---

## What's Next?

### Learn by Example

- [Statistics](https://observablehq.com/@essi/stats) - t-tests, ANOVA, GLM, mixed-effects models
- [Ordination](https://observablehq.com/@essi/ordinations) - PCA, LDA, RDA for multivariate analysis
- [Clustering](https://observablehq.com/@essi/clustering) - Hierarchical and k-means clustering
- [Machine learning](https://observablehq.com/@essi/machine-learning) - Classification and regression

### Browse the API

- [Statistics API](api/statistics)
- [Machine learning API](api/machine-learning)
- [Multivariate analysis API](api/multivariate)
- [Visualization API](api/visualization)

### Run Interactive Examples

Check out the [Examples](examples) page with live, runnable code snippets.

---

## Need Help?

- **Documentation**: Browse the [Tutorials](tutorials) and [API Reference](api)
- **GitHub Issues**: [Report bugs or request features](https://github.com/tangent-to/ds/issues)
- **Discussions**: [Ask questions](https://github.com/tangent-to/ds/discussions)

---

## Development Setup

Want to contribute? Clone the repository and install dependencies:

```bash
git clone https://github.com/tangent-to/ds.git
cd ds
npm install

# Run tests
npm test

# Build
npm run build
```

See [CONTRIBUTING.md](https://github.com/tangent-to/ds/blob/main/CONTRIBUTING.md) for more details.
