---
layout: home
title: Home
nav_order: 1
description: "tangent/ds is a data science toolkit for JavaScript/TypeScript, bringing statistical analysis and machine learning to the browser and Node.js/Deno."
permalink: /
---

# Data science for JavaScript
{: .fs-9 }

Analyze data, build models, visualize results—all in JavaScript.
{: .fs-6 .fw-300 }

[Get Started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/tangent-to/ds){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Quick example

```javascript
import * as ds from '@tangent/ds';
import * as Plot from '@observablehq/plot';

// Run PCA in 3 lines
const pca = new ds.mva.PCA({ center: true, scale: true });
pca.fit({ data: myData, X: ["var1", "var2", "var3"] });
ds.plot.ordiplot(pca.model).show(Plot);
```

---

## Features

### Multivariate analysis
Ordination techniques (PCA, LDA, RDA) and clustering algorithms (hierarchical, k-means) for exploring complex datasets.

[Try Ordination](https://observablehq.com/@essi/ordinations){: .btn .btn-outline }

### Statistical analysis
Comprehensive statistical toolkit including t-tests, ANOVA, generalized linear models (GLM), and mixed-effects models.

[Explore Statistics](https://observablehq.com/@essi/stats){: .btn .btn-outline }

### Machine learning
Build predictive models with K-nearest neighbors, multilayer perceptrons, decision trees, and more. Complete with cross-validation and hyperparameter tuning.

[Learn ML](https://observablehq.com/@essi/machine-learning){: .btn .btn-outline }

### Visualization
Publication-ready plots powered by Observable Plot. Create biplots, confusion matrices, dendrograms, and more.

[See Examples](examples){: .btn .btn-outline }

---

## Why tangent/ds?

**Native JavaScript**
: No Python or R required. Run analyses directly in the browser or Node.js/Deno.

**Modern API**
: Clean, intuitive API inspired by scikit-learn and R's tidyverse.

**Toolkit**
: From data preprocessing to model evaluation, in one package.

---

## Installation

```bash
# npm
npm install @tangent/ds

# deno
deno add @tangent/ds
```

[Read the full guide →](getting-started)

---

## Community

- **GitHub**: [tangent-to/ds](https://github.com/tangent-to/ds)
- **Issues**: [Report bugs](https://github.com/tangent-to/ds/issues)
- **Discussions**: [Ask questions](https://github.com/tangent-to/ds/discussions)

---

## License

tangent/ds is distributed under the [GPL-3 License](https://github.com/tangent-to/ds/blob/main/LICENSE).
