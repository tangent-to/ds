/**
 * @tangent.to/ds - A minimalist, browser-friendly data-science library
 * 
 * Main entry point exporting core, stats, ml, mva, and plot namespaces
 */

/** Core numerical primitives (data frames, linear algebra, math helpers). */
import * as core from './core/index.js';
/** Statistics: distributions, hypothesis tests, GLMs, model comparison. */
import * as stats from './stats/index.js';
/** Machine-learning estimators (regression, classification, etc.). */
import * as ml from './ml/index.js';
/** Multivariate analysis (PCA, factor analysis, and related methods). */
import * as mva from './mva/index.js';
/** Plotting utilities for browser-friendly SVG/canvas charts. */
import * as plot from './plot/index.js';

export { core, stats, ml, mva, plot };

/**
 * Default export bundling every namespace (core, stats, ml, mva, plot) under
 * one object for convenient single-import usage.
 */
export default { core, stats, ml, mva, plot };
