---
title: 'Data Science for JavaScript with @tangent.to/ds'
tags:
  - JavaScript
  - data science
  - statistics
  - machine learning
  - multivariate analysis
  - Observable
authors:
  - name: Essi Parent
    orcid: 0000-0003-1679-2287
    affiliation: "1, 2"
    corresponding: true
affiliations:
  - index: 1
    name: Département des sols et du génie agroalimentaire, Université Laval, Québec, Canada
    ror: "04sjchr03"
  - index: 2
    name: Ministère de l'Environnement, de la Lutte contre les changements climatiques, de la Faune et des Parcs, Québec, Canada
date: 9 February 2026
bibliography: paper.bib
---

# Summary
tangent.to/ds is a JavaScript library that provides a unified data science toolkit for browsers and server-side JavaScript environments. It implements multivariate analysis methods, data clustering, statistical modeling, machine learning, and in pure ESM JavaScript, filling a gap in the JavaScript ecosystem where data wrangling and visualization tools exist but statistical and analytical capabilities remain fragmented or absent. The library offers five integrated modules: `core` (linear algebra, optimization, formula parsing), `stats` (distributions, generalized linear models, hypothesis tests), `ml` (clustering, classification, regression, neural networks, cross-validation), `mva` (principal component analysis, linear discriminant analysis, redundancy analysis, canonical correspondence analysis, hierarchical clustering), and `plot` (Observable Plot configuration generators for diagnostic and ordination plots).

# Statement of need
The JavaScript data science ecosystem already has mature libraries for data wrangling and visualization. ml-matrix [@ml-matrix] provides fast matrix operations, simple statistics [@simplestats] a collection of statistical testing, Arquero [@arquero]  and Danfojs [@danfojs] dplyr-inspired tabular transformations, as well as Observable Plot [@observableplot] a concise, high-level visualization. However, a critical gap remains: none of these tools provide a comprehensive, one-stop API for statistical modeling, hypothesis testing, or multivariate analysis. Typically, a data scientist working in an Observable notebook can import, clean, reshape, and plot data entirely in JavaScript, leave the browser environment to an R or Python environment fit a generalized linear model, run a PCA, or perform an ANOVA, and either postprocess data away from JavaScript, either export everything to get back to Observable for another round of visualization. tangent.to/ds eliminates this context-switching by providing, directly in JavaScript, R-level statistical [@rbase] and multivariate data analyses [@rvegan], as well as Scikit-Learn-level [@sklearn] machine learning.

The library targets four audiences: (1) students, self-learners and teachers who need a reliable computing environment to learn and teach data science without complicated computer setups, (2) researchers using Observable notebooks in need of analytical methods beyond data wrangling, (3) developers building browser-based data applications that require embedded data computations without server round-trips, and (4) data scientists who prefer or require JavaScript environments (Deno, Node.js) for their analysis pipelines.

# State of the field
No existing JavaScript library offers the combination of GLM/GLMM fitting with formula syntax, statistical distributions with PDF/CDF/quantile functions, model comparison via AIC/BIC and likelihood ratio tests, ordination methods (PCA, LDA, RDA, CCA), and integrated diagnostic plotting, which together constitute the standard analytical workflow in R or Python. tangent.to/ds was built to fill this gap.

# Software design

tangent.to/ds is structured as five modules that mirror the conceptual organization of a data analysis workflow, from data manipulation (`core`) through statistical modeling (`stats`), machine learning (`ml`), multivariate analysis (`mva`), and visualization (`plot`).  API design decisions reflect the library's intended use in browser-based and Observable environments.

- **Minimal dependencies.** The library relies on well-established JavaScript packages (ml-matrix for linear algebra, simple-statistics for basic statistics and Arquero for data manipulation) and requires no WebAssembly compilation or bundler configuration.
- **Observable Plot integration.** The `plot` module generates configuration objects for Observable Plot rather than rendering graphics directly. This design composes naturally with Observable notebooks, where the user calls `Plot.plot(config)` to render, and avoids coupling the library to a specific rendering backend.
- **Tidy and formula syntax**. The API is inspired by Scikit-Learn's pipeline approach ([@sklearn]), and by tidyverse ([@tidyverse]) chaining ethos. The GLM class accepts Wilkinson-Rogers formula notation (`y ~ x1 + x2`), lowering the barrier for researchers transitioning from R. Formulas are parsed at fit time and resolved against a data object, enabling a concise and familiar modeling interface.
- **Compositional data awareness.** Often overlooked but of important impact in data science, log-ratio transformations for compositional variables are implemented in the core module, reflecting the author's research in geosciences and ecology where compositional data analysis [@pawlowsky2015] is routine but rarely supported in general-purpose libraries.
- **Numerical validation against R and Python.** The test suite includes cross-language validation tests (`tests_compare-to-R/`, `tests_compare-to-python/`) that verify numerical outputs against R's base packages ([@rbase]), R's vegan package ([@rvegan]) and Scikit-Learn ([@sklearn]), ensuring that results are reproducible across ecosystems.

For example, fitting a linear model uses a syntax familiar to R users.

```javascript
import { GLM } from "@tangent.to/ds";
const simple_lm = new GLM({ family: "gaussian", link: "identity" })
  .fit('`Body Mass (g)` ~ `Beak Length (mm)`', penguinsData);
console.log(simple_lm.summary());
```

# Research impact statement

tangent.to/ds is currently used as lecture notes in the Observable collection [Data science with `tangent/ds`](https://observablehq.com/collection/@essi/data-science-with-tangent), and by the author in research in agroenvironmental science at Université Laval. The library enables fully browser-based analytical workflows in Observable notebooks for research applications including soil compaction modeling and compositional analysis of environmental data.

The library is published on npm as `@tangent.to/ds` and on JSR as `@tangent/ds`, with documentation hosted at [tangent-to.github.io/ds/](https://tangent-to.github.io/ds/). The repository includes working examples, a complete API reference, and cross-language validation tests. While the library is at an early stage of community adoption, it addresses a documented gap in the JavaScript ecosystem [@js4ds] and is designed for immediate use in Observable notebooks, one of the most widely adopted platforms for browser-based data analysis.

# AI usage disclosure
Claude Code was used during development for code generation assistance and debugging. All AI-generated code was reviewed, tested, and validated against R and Python reference implementations. This paper was reviewed by AI for language and references. All scientific content and claims were authored and verified by the author.

# References
