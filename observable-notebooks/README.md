# Observable Notebooks for tangent-ds

This directory contains four comprehensive, beginner-friendly Observable notebooks showcasing tangent-ds capabilities.

## 📚 Notebooks Overview

### 1. **Ordinations** (`01-ordinations.md`)
**Topics covered**:
- Principal Component Analysis (PCA)
- Correspondence Analysis (CA)
- Redundancy Analysis (RDA)
- Unified ordination plotting
- Biplots, scree plots, variance partitioning

**Datasets used**:
- `vega-datasets.iris` - Classic iris flower measurements
- `vega-datasets.cars` - Automotive data
- Custom movie preference data

**Key features**:
- Beautiful visualizations with convex hulls
- Interactive tooltips
- Clear explanations of eigenvalues and variance
- Best practices for choosing ordination methods

---

### 2. **Clustering** (`02-clustering.md`)
**Topics covered**:
- Hierarchical Clustering (HCA)
- K-Means Clustering
- Elbow method for choosing k
- Silhouette analysis
- Dendrograms and cluster visualization

**Datasets used**:
- `vega-datasets.iris` - Species clustering
- Simulated customer segmentation data
- Gapminder world development data

**Key features**:
- Interactive dendrograms
- Customer profiling examples
- Real-world country clustering
- Comparison of clustering methods
- Visual guide to interpreting results

---

### 3. **Statistics** (`03-statistics.md`)
**Topics covered**:
- Hypothesis testing (t-tests, ANOVA)
- Linear regression with R-style formulas
- Generalized Linear Models (GLM)
  - Gaussian (normal)
  - Binomial (logistic)
  - Poisson (count data)
  - Gamma, Inverse Gaussian, Negative Binomial
- Mixed effects models (GLMM)
- Model diagnostics
- Model comparison (AIC, BIC, likelihood ratio tests)
- Cross-validation

**Datasets used**:
- `vega-datasets.cars` - Automotive analysis
- Simulated sales and student data

**Key features**:
- **NEW**: R-style formula syntax (`y ~ x1 + x2`)
- **NEW**: Diagnostic plots (residual, QQ, scale-location)
- **NEW**: Model comparison utilities
- Beautiful regression visualizations
- Mixed models with random effects
- Complete statistical workflow

---

### 4. **Machine Learning** (`04-machine-learning.md`)
**Topics covered**:
- Binary classification
- Multiclass classification
- Regression
- Model evaluation (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves and AUC
- Precision-Recall curves
- Decision boundaries
- Polynomial regression
- Cross-validation
- Learning curves
- Feature importance

**Datasets used**:
- `vega-datasets.iris` - Classification tasks
- `vega-datasets.cars` - Regression and classification
- Simulated data for demonstrations

**Key features**:
- Train/test splitting
- Beautiful decision boundary plots
- Complete ML pipeline examples
- Visual model evaluation
- Practical best practices
- Real-world examples

---

## 🚀 How to Use These Notebooks

### Option 1: Copy-Paste into Observable

1. Go to [Observable](https://observablehq.com)
2. Create a new notebook (or open your existing notebook)
3. Copy the entire markdown content from any `.md` file
4. Paste it into Observable
5. Observable will automatically parse the markdown and create interactive cells!

### Option 2: Import Directly

If you publish these as Observable notebooks, they can be imported:

```js
import {pca, kmeans} from "@your-username/notebook-name"
```

---

## 📝 Observable Markdown Features Used

These notebooks use Observable's extended markdown syntax:

### Code Cells
```js
// JavaScript code that runs and can be referenced
x = 5
```

### Reactive Cells
```js
// This cell automatically updates when dependencies change
y = x * 2
```

### Display Markdown with Computed Values
```js
md`The value is ${x}`
```

### LaTeX Math
Inline: `$E = mc^2$` becomes $E = mc^2$

Display mode:
```
$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

### HTML Blocks
```js
html`<div style="color: red;">Custom HTML</div>`
```

### Inputs
```js
Inputs.table(data)
Inputs.range([0, 100])
Inputs.select(["A", "B", "C"])
```

---

## 🎨 Design Principles

All notebooks follow these principles for learners:

1. **📖 Clear Explanations**: Every concept explained in plain language
2. **🎯 Real Examples**: Using familiar, real-world datasets
3. **📊 Beautiful Visuals**: Every example has an attractive plot
4. **💡 Interpretations**: Always explain what results mean
5. **🎓 Exercises**: Hands-on practice problems
6. **⚡ Interactive**: Users can modify and experiment
7. **🔗 Resources**: Links to learn more

---

## 🎨 Visual Style Guide

### Color Schemes Used
- **Observable Plot defaults**: For consistency
- **tableau10**: For categorical data
- **purples/blues**: For continuous scales
- **Semantic colors**:
  - ✅ Green = good/positive
  - ❌ Red = bad/negative
  - 🔵 Blue = neutral/data
  - 🟡 Yellow/Gold = highlights

### Plot Features
- Grid lines enabled for reference
- Tooltips (`tip: true`) for exploration
- Clear axis labels with units
- Captions explaining what to look for
- Legends when needed
- Reference lines (means, zeros, diagonals)

---

## 🔧 Dependencies

All notebooks require:
```js
import * as ds from "@tangent.to/ds"
import * as Plot from "@observablehq/plot"
import * as vega from "vega-datasets"
```

Optional but recommended:
```js
import * as d3 from "d3"  // Usually auto-available in Observable
import * as Inputs from "@observablehq/inputs"  // For interactive controls
```

---

## 📊 Dataset Sources

| Dataset | Source | Use Cases |
|---------|--------|-----------|
| `iris` | vega-datasets | Classification, clustering, ordination |
| `cars` | vega-datasets | Regression, prediction, clustering |
| `gapminder` | GitHub raw | Country clustering, correlation |
| Custom simulated | Generated in notebook | Specific teaching examples |

---

## 🎓 Target Audience

These notebooks are designed for:
- **Data science students** learning statistics and ML
- **Researchers** new to computational analysis
- **Observable users** exploring tangent-ds
- **R/Python users** transitioning to JavaScript
- **Anyone** curious about data analysis!

**Prerequisites**:
- Basic JavaScript (variables, functions, arrays)
- Basic statistics (mean, variance, correlation)
- Curiosity! 🎉

---

## 🚀 Future Enhancements

Potential additions:
- [ ] Time series analysis notebook
- [ ] Bayesian statistics notebook
- [ ] Text analysis notebook
- [ ] Network analysis notebook
- [ ] Advanced deep learning notebook
- [ ] Survival analysis notebook

---

## 🤝 Contributing

Found an error? Have an improvement?
1. Open an issue on [tangent-ds GitHub](https://github.com/tangent-to/tangent-ds)
2. Suggest edits via pull request
3. Share your own notebook examples!

---

## 📜 License

These notebooks are released under the same license as tangent-ds (MIT).

Feel free to:
- ✅ Use in your own teaching
- ✅ Adapt for your domain
- ✅ Share with attribution
- ✅ Build upon them

---

## 🌟 Showcase Your Work!

Built something cool with these notebooks? Share it!
- Tweet with `#tangentds` and `#observablehq`
- Link in GitHub discussions
- Submit to Observable's trending notebooks

---

*Created with ❤️ for the Observable and data science communities*
