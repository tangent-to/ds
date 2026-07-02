import { attachShow } from './show.js';
import { resolveGroupValues } from './utils.js';

/**
 * Unified ordination plot (ordiplot) for PCA, LDA, and RDA
 * Provides a consistent interface for plotting ordination results
 */

/**
 * Generate unified ordination plot configuration
 * Works with PCA, LDA, and RDA results
 *
 * @param {Object} result - Ordination result (from PCA, LDA, or RDA)
 * @param {Object} options - Configuration options
 * @param {string} options.type - Type of ordination ('pca', 'lda', 'rda') - auto-detected if not specified
 * @param {Array|Iterable|Object|string|null} options.colorBy - Group values for points: an array,
 *   any iterable (e.g. an Arquero column), a { data, column } descriptor, or the name of a
 *   column in the data the model was fit on (requires a declarative fit({ data, ... }))
 * @param {Array|Iterable|Object|string|null} options.labels - Labels for points (same accepted forms as colorBy)
 * @param {boolean} options.showLoadings - Show loading vectors (PCA/RDA only)
 * @param {boolean} options.showCentroids - Show class centroids (LDA only)
 * @param {boolean} options.showConvexHulls - Show convex hulls around groups (optional)
 * @param {number} options.axis1 - First axis to plot (default: 1)
 * @param {number} options.axis2 - Second axis to plot (default: 2)
 * @param {number} options.width - Plot width (default: 640)
 * @param {number} options.height - Plot height (default: 400)
 * @param {number} options.loadingScale - Scale factor for loading vectors (default: 3)
 * @param {number} options.loadingFactor - Multiplier applied to loading vectors (default: 1, set 0 for auto)
 * @param {number|null} options.predictorFactor - Multiplier for predictor arrows (RDA only, default: inherits loadingFactor; set 0 for auto)
 * @param {Object|null} options.color - Point color scale for grouped points, merged
 *   into the Plot `color` scale: `{ range, scheme, domain, legend, label }`. E.g.
 *   `{ range: ["#111","#888"] }` for greyscale, or `{ scheme: "Observable10" }`.
 * @param {string} options.pointColor - Fill for points when there is no `colorBy`
 *   (default: 'steelblue').
 * @param {boolean} options.symbolBy - Also encode groups by symbol (shape), not
 *   colour alone - recommended for greyscale / colour-vision-deficiency safety.
 * @param {string|null} options.loadingColor - Stroke for loading arrows (default:
 *   'red', or 'blue' for an RDA response triplot).
 * @param {string|null} options.loadingTextColor - Fill for loading labels (default:
 *   'darkred', or 'darkblue' for RDA).
 * @param {number} options.pointRadius - Point radius (default: 4).
 * @returns {Object} Plot configuration
 */
export function ordiplot(result, {
  type = null,
  colorBy = null,
  labels = null,
  showLoadings = true,
  showCentroids = false,
  showConvexHulls = false,
  axis1 = 1,
  axis2 = 2,
  width = 640,
  height = 400,
  loadingScale = 3,
  loadingFactor = 1,
  predictorFactor = null,
  color = null,
  pointColor = 'steelblue',
  symbolBy = false,
  loadingColor = null,
  loadingTextColor = null,
  pointRadius = 4
} = {}) {
  // Auto-detect ordination type if not specified
  if (!type) {
    type = detectOrdinationType(result);
  }

  // Accept arrays, iterables (e.g. Arquero columns), { data, column }
  // descriptors, or a column name from the model's source data
  colorBy = resolveGroupValues(colorBy, result, 'colorBy');
  labels = resolveGroupValues(labels, result, 'labels');

  // Extract scores and construct data based on ordination type
  const { scoresData, loadingsData, centroidsData, axisLabels, predictorData } = extractOrdinationData(
    result,
    type,
    axis1,
    axis2,
    colorBy,
    labels,
    loadingScale
  );

  if (colorBy && colorBy.length !== scoresData.length) {
    throw new Error(
      `colorBy has ${colorBy.length} values but the ordination has ${scoresData.length} scores. ` +
        `If rows with missing values were dropped during fitting (naOmit), ` +
        `use a column name or { data, column } so values stay aligned, ` +
        `or filter your colorBy array the same way.`,
    );
  }

  // Build plot configuration
  const config = {
    type: 'ordiplot',
    ordinationType: type,
    width,
    height,
    data: {
      scores: scoresData
    },
    axes: {
      x: { label: axisLabels.x, grid: true },
      y: { label: axisLabels.y, grid: true }
    },
    marks: []
  };

  // Add convex hulls if requested (must be added first so they're behind points)
  if (showConvexHulls && colorBy) {
    const hullData = computeConvexHulls(scoresData, colorBy);
    if (hullData.length > 0) {
      config.data.hulls = hullData;
      config.marks.push({
        type: 'area',
        data: 'hulls',
        x: 'x',
        y: 'y',
        fill: 'group',
        fillOpacity: 0.2,
        stroke: 'group',
        strokeWidth: 1
      });
    }
  }

  // Add score points. Group identity is carried by a fill colour scale and,
  // when symbolBy is set, also by shape - so the plot stays readable in greyscale
  // and under colour-vision deficiency.
  // Real groups only: without colorBy every point carries color:'default', which
  // should render as one constant fill (pointColor), not a one-entry colour scale.
  const hasColorField = Array.isArray(colorBy) && colorBy.length > 0 &&
    scoresData.length > 0 && scoresData.every((d) => typeof d.color !== 'undefined');
  const dotMark = {
    type: 'dot',
    data: 'scores',
    x: 'x',
    y: 'y',
    fill: hasColorField ? 'color' : pointColor,
    r: pointRadius,
    fillOpacity: 0.7,
    tip: labels ? true : false
  };
  if (hasColorField && symbolBy) dotMark.symbol = 'color';
  config.marks.push(dotMark);

  // Expose the colour (and optional symbol) scale so `.show(Plot)` renders a
  // legend by default; the `color` option merges in range/scheme/domain overrides.
  if (hasColorField) {
    config.color = { legend: true, ...(color || {}) };
    if (symbolBy) config.symbol = { legend: true };
  }

  // Add labels if provided
  if (labels) {
    config.marks.push({
      type: 'text',
      data: 'scores',
      x: 'x',
      y: 'y',
      text: 'label',
      fontSize: 10,
      dx: 8,
      dy: -8
    });
  }

  // Add loadings for PCA, LDA, and RDA
  if (showLoadings && loadingsData && (type === 'pca' || type === 'lda' || type === 'rda')) {
    const computeFactor = (vectors, requestedFactor) => {
      let factor = requestedFactor ?? 1;
      if (factor === 0) {
        const maxScoreRadius = scoresData.reduce(
          (max, point) => Math.max(max, Math.hypot(point.x || 0, point.y || 0)),
          0
        );
        const maxVectorRadius = vectors.reduce(
          (max, vec) => Math.max(max, Math.hypot(vec.x2 || vec.dx || 0, vec.y2 || vec.dy || 0)),
          0
        );
        if (maxVectorRadius > 0) {
          factor = maxScoreRadius > 0 ? maxScoreRadius / maxVectorRadius : 1;
        } else {
          factor = 1;
        }
      }
      return factor;
    };

    const appliedLoadingFactor = computeFactor(loadingsData, loadingFactor);
    const scaledLoadings = appliedLoadingFactor === 1
      ? loadingsData
      : loadingsData.map((loading) => ({
          ...loading,
          x2: (loading.x2 || 0) * appliedLoadingFactor,
          y2: (loading.y2 || 0) * appliedLoadingFactor
        }));

    config.data.loadings = scaledLoadings;

    if (type === 'rda' && predictorData && predictorData.length) {
      const appliedPredictorFactor = computeFactor(
        predictorData,
        predictorFactor === null ? loadingFactor : predictorFactor
      );
      const scaledPredictors = appliedPredictorFactor === 1
        ? predictorData
        : predictorData.map((pred) => ({
            ...pred,
            x2: (pred.x2 || 0) * appliedPredictorFactor,
            y2: (pred.y2 || 0) * appliedPredictorFactor
          }));
      config.data.predictors = scaledPredictors;
    }

    // Loading colours: honour explicit overrides, else default (RDA response = blue,
    // otherwise red). Pass e.g. loadingColor: "#111" for a greyscale biplot.
    const loadingStroke = loadingColor ?? ((type === 'rda' && predictorData) ? 'blue' : 'red');
    const loadingLabelFill = loadingTextColor ?? ((type === 'rda' && predictorData) ? 'darkblue' : 'darkred');

    config.marks.push({
      type: 'arrow',
      data: 'loadings',
      x1: 'x1',
      y1: 'y1',
      x2: 'x2',
      y2: 'y2',
      stroke: loadingStroke,
      strokeWidth: 2,
      headLength: 8
    });

    // Label placement: anchor each label on the outward side of its arrow tip and
    // spread labels vertically so near-collinear vectors don't stack on top of
    // each other (the common failure mode of dense biplots). A white halo keeps
    // labels legible where they cross arrows or points.
    const placed = placeLoadingLabels(scaledLoadings);
    config.data.loadingLabelsRight = placed.filter((d) => d.side === 'right');
    config.data.loadingLabelsLeft = placed.filter((d) => d.side === 'left');

    const labelMark = (dataKey, anchor, dx) => ({
      type: 'text',
      data: dataKey,
      x: 'x2',
      y: 'ly',
      text: 'variable',
      fontSize: 10,
      fontWeight: 600,
      fill: loadingLabelFill,
      stroke: 'white',
      strokeWidth: 3,
      textAnchor: anchor,
      dx,
      dy: 0
    });
    config.marks.push(labelMark('loadingLabelsRight', 'start', 6));
    config.marks.push(labelMark('loadingLabelsLeft', 'end', -6));
  }

  // Add predictor correlations for RDA triplot
  if (predictorData && type === 'rda') {
    if (!config.data.predictors) {
      config.data.predictors = predictorData;
    }
    config.marks.push({
      type: 'arrow',
      data: 'predictors',
      x1: 'x1',
      y1: 'y1',
      x2: 'x2',
      y2: 'y2',
      stroke: 'red',
      strokeWidth: 2,
      headLength: 8
    });
    config.marks.push({
      type: 'text',
      data: 'predictors',
      x: 'x2',
      y: 'y2',
      text: 'variable',
      fontSize: 10,
      fill: 'darkred',
      dx: 5,
      dy: 5
    });
  }

  // Add centroids for LDA
  if (showCentroids && centroidsData && type === 'lda') {
    config.data.centroids = centroidsData;
    config.marks.push({
      type: 'dot',
      data: 'centroids',
      x: 'x',
      y: 'y',
      fill: 'color',
      r: 8,
      stroke: 'black',
      strokeWidth: 2,
      symbol: 'cross'
    });
    config.marks.push({
      type: 'text',
      data: 'centroids',
      x: 'x',
      y: 'y',
      text: 'class',
      fontSize: 12,
      fontWeight: 'bold',
      dy: -15
    });
  }

  return attachShow(config);
}

/**
 * Place loading labels so near-collinear vectors don't overlap.
 *
 * Labels are anchored on the outward (left/right) side of each arrow tip, then
 * spread apart vertically within each side using a greedy minimum-gap pass. The
 * gap is a fraction of the largest loading radius, so it scales with the plot.
 * Returns each loading augmented with `ly` (label y) and `side`.
 * @private
 */
function placeLoadingLabels(loadings) {
  const maxRadius = loadings.reduce(
    (max, l) => Math.max(max, Math.hypot(l.x2 || 0, l.y2 || 0)),
    0
  );
  const gap = (maxRadius || 1) * 0.14;

  const spread = (group) => {
    group.sort((a, b) => (b.y2 || 0) - (a.y2 || 0)); // top to bottom
    let prev = Infinity;
    return group.map((d) => {
      let ly = d.y2 || 0;
      if (prev - ly < gap) ly = prev - gap;
      prev = ly;
      return { ...d, ly };
    });
  };

  const right = spread(loadings.filter((d) => (d.x2 || 0) >= 0).map((d) => ({ ...d, side: 'right' })));
  const left = spread(loadings.filter((d) => (d.x2 || 0) < 0).map((d) => ({ ...d, side: 'left' })));
  return [...right, ...left];
}

/**
 * Detect ordination type from result structure
 * @private
 */
function detectOrdinationType(result) {
  // Check for LDA first (most specific - has 'class' field in scores)
  if (result.scores && result.scores[0] && 'class' in result.scores[0]) {
    return 'lda';
  }
  // Check for RDA (has canonical scores/loadings)
  if (result.canonicalScores && result.canonicalLoadings) {
    return 'rda';
  }
  // Check for PCA (has scores, loadings, eigenvalues)
  if (result.scores && result.loadings && result.eigenvalues) {
    return 'pca';
  }
  throw new Error('Cannot detect ordination type. Please specify type option.');
}

/**
 * Extract and format ordination data
 * @private
 */
function extractOrdinationData(result, type, axis1, axis2, colorBy, labels, loadingScale) {
  let scoresData = [];
  let loadingsData = null;
  let centroidsData = null;
  let axisLabels = { x: '', y: '' };

  if (type === 'pca') {
    const { scores, loadings } = result;

    scoresData = scores.map((score, i) => ({
      x: score[`pc${axis1}`],
      y: score[`pc${axis2}`],
      index: i,
      color: colorBy ? colorBy[i] : 'default',
      label: labels ? labels[i] : `${i}`
    }));

    if (loadings) {
      loadingsData = loadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`pc${axis1}`] * loadingScale,
        y2: loading[`pc${axis2}`] * loadingScale,
        variable: loading.variable || `Var${i + 1}`,
        index: i
      }));
    }

    axisLabels = {
      x: `PC${axis1}`,
      y: `PC${axis2}`
    };

  } else if (type === 'lda') {
    const { scores, loadings, classMeanScores } = result;

    // Check if we have both axes
    const hasAxis2 = scores[0] && scores[0][`ld${axis2}`] !== undefined;

    scoresData = scores.map((score, i) => ({
      x: score[`ld${axis1}`] || 0,
      y: hasAxis2 ? score[`ld${axis2}`] : 0,
      index: i,
      color: score.class || (colorBy ? colorBy[i] : 'default'),
      label: labels ? labels[i] : `${i}`,
      class: score.class
    }));

    // Extract loadings if available
    if (loadings) {
      loadingsData = loadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`ld${axis1}`] * loadingScale,
        y2: hasAxis2 && loading[`ld${axis2}`] !== undefined ? loading[`ld${axis2}`] * loadingScale : 0,
        variable: loading.variable || `Var${i + 1}`,
        index: i
      }));
    }

    // Extract centroids if available
    if (classMeanScores && Array.isArray(classMeanScores)) {
      const classes = [...new Set(scores.map(s => s.class))];
      centroidsData = classMeanScores.map((means, i) => {
        const meanArray = Array.isArray(means) ? means : [means];
        return {
          x: meanArray[axis1 - 1] || 0,
          y: hasAxis2 && meanArray[axis2 - 1] !== undefined ? meanArray[axis2 - 1] : 0,
          class: classes[i] || `Class ${i}`,
          color: classes[i] || `Class ${i}`
        };
      });
    }

    axisLabels = {
      x: `LD${axis1}`,
      y: `LD${axis2}`
    };

  } else if (type === 'rda') {
    const { canonicalScores, canonicalLoadings, predictorCorrelations } = result;

    scoresData = canonicalScores.map((score, i) => ({
      x: score[`rda${axis1}`],
      y: score[`rda${axis2}`],
      index: i,
      color: colorBy ? colorBy[i] : 'default',
      label: labels ? labels[i] : `Site ${i}`
    }));

    // Response loadings (species/response variables)
    if (canonicalLoadings) {
      loadingsData = canonicalLoadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`rda${axis1}`] * loadingScale,
        y2: loading[`rda${axis2}`] * loadingScale,
        variable: loading.variable || `Var${i + 1}`,
        index: i,
        type: 'response'
      }));
    }

    // For triplot: use predictor correlations
    let predictorData = null;
    if (predictorCorrelations && predictorCorrelations.length > 0) {
      predictorData = predictorCorrelations.map((corr, i) => ({
        x1: 0,
        y1: 0,
        x2: corr[`rda${axis1}`] * loadingScale,
        y2: corr[`rda${axis2}`] * loadingScale,
        variable: corr.variable || `Pred${i + 1}`,
        index: i
      }));
    }

    axisLabels = {
      x: `RDA${axis1}`,
      y: `RDA${axis2}`
    };

    return { scoresData, loadingsData, centroidsData, axisLabels, predictorData };
  }

  return { scoresData, loadingsData, centroidsData, axisLabels };
}

/**
 * Compute convex hulls for grouped data
 * @private
 */
function computeConvexHulls(scoresData, colorBy) {
  // Group points by color
  const groups = new Map();
  scoresData.forEach((point, i) => {
    const group = colorBy[i];
    if (!groups.has(group)) {
      groups.set(group, []);
    }
    groups.get(group).push(point);
  });

  // Compute convex hull for each group
  const hullData = [];
  for (const [group, points] of groups.entries()) {
    if (points.length < 3) continue; // Need at least 3 points for a hull

    const hull = convexHull(points);
    hull.forEach(point => {
      hullData.push({
        x: point.x,
        y: point.y,
        group
      });
    });
  }

  return hullData;
}

/**
 * Compute convex hull using gift wrapping algorithm
 * @private
 */
function convexHull(points) {
  if (points.length < 3) return points;

  // Find leftmost point
  let leftmost = points[0];
  for (const p of points) {
    if (p.x < leftmost.x || (p.x === leftmost.x && p.y < leftmost.y)) {
      leftmost = p;
    }
  }

  const hull = [];
  let current = leftmost;

  do {
    hull.push(current);
    let next = points[0];

    for (const p of points) {
      if (next === current || isLeftTurn(current, next, p)) {
        next = p;
      }
    }

    current = next;
  } while (current !== leftmost && hull.length < points.length);

  return hull;
}

/**
 * Check if three points make a left turn
 * @private
 */
function isLeftTurn(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
}
