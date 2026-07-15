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
 * @param {number} options.minLoadingContribution - Hide loading/predictor vectors
 *   whose contribution to the two displayed axes is below this fraction (0-1) of
 *   the total, i.e. squared vector length / summed squared length. Default 0 shows
 *   every vector at its true relative length. Use e.g. 0.02 to drop near-zero
 *   vectors that only clutter the plot, keeping magnitudes honest (vectors are NOT
 *   rescaled - negligible ones are removed, not inflated).
 * @param {number} options.labelNudge - Multiplier on the radial distance a loading/
 *   predictor label sits past its arrow tip (default 1). Larger pushes labels
 *   further out from the tips before de-collision.
 * @param {boolean} options.labelRepel - Run ggrepel-style overlap removal on the
 *   arrow labels (default true). Set false to place each label at its tip with no
 *   de-collision.
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
 * @param {string|null} options.predictorColor - Stroke for RDA predictor arrows
 *   (default: 'red'). Set e.g. '#111' for a greyscale triplot.
 * @param {string|null} options.predictorTextColor - Fill for RDA predictor labels
 *   (default: inherits predictorColor, else 'darkred').
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
  minLoadingContribution = 0,
  labelNudge = 1,
  labelRepel = true,
  color = null,
  pointColor = 'steelblue',
  symbolBy = false,
  loadingColor = null,
  loadingTextColor = null,
  predictorColor = null,
  predictorTextColor = null,
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

  // Build plot configuration. Reserve horizontal margins so loading/predictor labels
  // near the frame edge (arrow tips reach ~the score-cloud radius) are not clipped;
  // biplots/triplots are label-heavy on the left and right in particular.
  const config = {
    type: 'ordiplot',
    ordinationType: type,
    width,
    height,
    marginLeft: 90,
    marginRight: 110,
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
    // Auto-scale (factor 0) fits the longest vector to ARROW_HEADROOM x the score-cloud
    // radius, so arrow tips stay inside the cloud and leave room for their labels rather
    // than landing on the frame edge (which clips the text).
    const ARROW_HEADROOM = 0.9;
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
          factor = maxScoreRadius > 0 ? ARROW_HEADROOM * maxScoreRadius / maxVectorRadius : 1;
        } else {
          factor = 1;
        }
      }
      return factor;
    };

    // Optionally hide vectors that barely contribute to the two displayed axes.
    // Contribution = squared vector length / summed squared length (a vector's
    // share of the represented structure on these axes). Removing negligible
    // vectors declutters the plot WITHOUT distorting the magnitudes of the ones
    // kept (they are not rescaled). Scaling is uniform, so contribution is
    // scale-invariant; filtering before or after scaling gives the same set.
    const filterByContribution = (vectors) => {
      if (!(minLoadingContribution > 0) || vectors.length === 0) return vectors;
      const sq = vectors.map((v) => (v.x2 || 0) ** 2 + (v.y2 || 0) ** 2);
      const total = sq.reduce((a, b) => a + b, 0);
      if (total <= 0) return vectors;
      return vectors.filter((_, i) => sq[i] / total >= minLoadingContribution);
    };

    const appliedLoadingFactor = computeFactor(loadingsData, loadingFactor);
    const scaledLoadings = filterByContribution(appliedLoadingFactor === 1
      ? loadingsData
      : loadingsData.map((loading) => ({
          ...loading,
          x2: (loading.x2 || 0) * appliedLoadingFactor,
          y2: (loading.y2 || 0) * appliedLoadingFactor
        })));

    config.data.loadings = scaledLoadings;

    if (type === 'rda' && predictorData && predictorData.length) {
      const appliedPredictorFactor = computeFactor(
        predictorData,
        predictorFactor === null ? loadingFactor : predictorFactor
      );
      const scaledPredictors = filterByContribution(appliedPredictorFactor === 1
        ? predictorData
        : predictorData.map((pred) => ({
            ...pred,
            x2: (pred.x2 || 0) * appliedPredictorFactor,
            y2: (pred.y2 || 0) * appliedPredictorFactor
          })));
      config.data.predictors = scaledPredictors;
    }

    // Loading colours: honour explicit overrides, else default (RDA response = blue,
    // otherwise red). Pass e.g. loadingColor: "#111" for a greyscale biplot.
    const loadingStroke = loadingColor ?? ((type === 'rda' && predictorData) ? 'blue' : 'red');
    const loadingLabelFill = loadingTextColor ?? ((type === 'rda' && predictorData) ? 'darkblue' : 'darkred');

    // Draw each arrow as a white "halo" underneath the coloured arrow so it stays
    // visible over the dense point cloud (arrows were being masked by the dots).
    // Two marks: a thick white stroke, then the coloured arrow on top.
    const arrowWithHalo = (dataKey, stroke) => {
      config.marks.push({
        type: 'arrow', data: dataKey, x1: 'x1', y1: 'y1', x2: 'x2', y2: 'y2',
        stroke: 'white', strokeWidth: 4.5, strokeOpacity: 0.9, headLength: 8
      });
      config.marks.push({
        type: 'arrow', data: dataKey, x1: 'x1', y1: 'y1', x2: 'x2', y2: 'y2',
        stroke, strokeWidth: 2, headLength: 8
      });
    };
    arrowWithHalo('loadings', loadingStroke);

    // Draw predictor (RDA triplot) arrows now too, so ALL labels can be de-collided
    // together against one another below.
    const predLabelFill = predictorTextColor ?? predictorColor ?? 'darkred';
    if (predictorData && type === 'rda') {
      if (!config.data.predictors) config.data.predictors = predictorData;
      arrowWithHalo('predictors', predictorColor ?? 'red');
    }

    // ggrepel-style label placement (pixel-space repulsion + spring to tip), run
    // over loadings AND predictors together so their labels do not overlap each
    // other. `fill` tags each label so one text mark can colour both sets.
    const labelCtx = {
      scoresData,
      width,
      height,
      margins: { l: config.marginLeft, r: config.marginRight, t: 40, b: 40 },
      fontSize: 10,
      nudge: labelNudge,
      repel: labelRepel
    };
    const arrowLabels = [
      ...scaledLoadings.map((d) => ({ ...d, fill: loadingLabelFill })),
      ...(predictorData && type === 'rda'
        ? (config.data.predictors).map((d) => ({ ...d, fill: predLabelFill }))
        : [])
    ];
    const placed = repelLabels(arrowLabels, labelCtx);
    config.data.arrowLabels = placed;
    config.data.arrowLeaders = placed.filter((d) => d.leader);

    // Thin connector from tip to a drifted label (only when it moved far).
    if (config.data.arrowLeaders.length > 0) {
      config.marks.push({
        type: 'link', data: 'arrowLeaders',
        x1: 'tipx', y1: 'tipy', x2: 'lx', y2: 'ly',
        stroke: '#999', strokeWidth: 0.6
      });
    }
    // One text mark for all labels; white halo keeps them legible over arrows/points.
    config.marks.push({
      type: 'text', data: 'arrowLabels', x: 'lx', y: 'ly', text: 'variable',
      fontSize: 10, fontWeight: 600, fill: 'fill',
      stroke: 'white', strokeWidth: 3
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
 * ggrepel-style label placement for arrow (loading/predictor) labels.
 *
 * Each label starts at its arrow TIP nudged slightly outward, then a short
 * force simulation in PIXEL space pushes overlapping labels apart while a weak
 * spring pulls each back toward its own tip. This de-clutters dense biplots
 * (many arrows converging near the origin) without detaching a label from its
 * vector. When a label had to travel far from its tip, `leader` is set so the
 * caller can draw a thin connector line back to the arrow.
 *
 * Repulsion needs pixel geometry (label width/height depend on text length and
 * font size), so anchors are mapped data->pixels using the plot's drawing area,
 * de-collided, then mapped back to data coordinates.
 *
 * @param {Array} anchors - [{ x2, y2, variable, group, fill }] arrow tips (data coords)
 * @param {Object} ctx - { scoresData, width, height, margins:{l,r,t,b}, fontSize }
 * @returns {Array} each anchor + { lx, ly, tipx, tipy, side, leader }
 * @private
 */
function repelLabels(anchors, ctx) {
  if (anchors.length === 0) return [];
  const { scoresData, width, height, margins, fontSize, nudge: nudgeMul = 1, repel = true } = ctx;
  const plotW = Math.max(1, width - margins.l - margins.r);
  const plotH = Math.max(1, height - margins.t - margins.b);

  // Approximate the plot's data domain from the point cloud and the arrow tips
  // (Plot auto-fits to include all marks). Exactness is not required: only the
  // relative pixel geometry of the labels matters for de-collision.
  const xs = [0], ys = [0];
  for (const p of scoresData) { xs.push(p.x || 0); ys.push(p.y || 0); }
  for (const a of anchors) { xs.push(a.x2 || 0); ys.push(a.y2 || 0); }
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmax === xmin) { xmax += 1; xmin -= 1; }
  if (ymax === ymin) { ymax += 1; ymin -= 1; }
  const sx = plotW / (xmax - xmin);
  const sy = plotH / (ymax - ymin);
  const toPx = (x, y) => ({ px: (x - xmin) * sx, py: (ymax - y) * sy }); // y axis points up
  const toData = (px, py) => ({ x: px / sx + xmin, y: ymax - py / sy });

  // Label boxes in pixels. Each has:
  //   ax, ay  - arrow tip (the thing the label names) in px
  //   rx, ry  - REST position: tip nudged radially outward. The spring pulls here
  //             (not to the tip), so labels settle just past their arrowhead.
  //   px, py  - current label CENTRE in px (starts at rest).
  const origin = toPx(0, 0);
  const gap = 3;                                       // min pixel gap between boxes
  const labels = anchors.map((a) => {
    const tip = toPx(a.x2 || 0, a.y2 || 0);
    const w = Math.max(fontSize, String(a.variable || '').length * fontSize * 0.62) + 6;
    const h = fontSize * 1.3;
    let dx = tip.px - origin.px, dy = tip.py - origin.py;
    const d = Math.hypot(dx, dy) || 1;
    const nudge = (h * 0.6 + 2) * nudgeMul;             // clears the arrowhead; scaled by labelNudge
    const rx = tip.px + (dx / d) * nudge;
    const ry = tip.py + (dy / d) * nudge;
    return { ...a, w, h, ax: tip.px, ay: tip.py, rx, ry, px: rx, py: ry };
  });

  // Force simulation. Repulsion RESOLVES box overlaps (dominant); a WEAK spring
  // pulls each label back toward its rest point so it stays near its own arrow.
  const ITER = repel ? 300 : 0;
  for (let it = 0; it < ITER; it++) {
    // 1) resolve pairwise overlaps (minimum translation on the smaller axis)
    for (let i = 0; i < labels.length; i++) {
      for (let j = i + 1; j < labels.length; j++) {
        const A = labels[i], B = labels[j];
        const ox = (A.w + B.w) / 2 + gap - Math.abs(A.px - B.px);
        const oy = (A.h + B.h) / 2 + gap - Math.abs(A.py - B.py);
        if (ox > 0 && oy > 0) {
          if (ox < oy) {
            const s = (ox / 2 + 0.5) * (A.px <= B.px ? 1 : -1);
            A.px -= s; B.px += s;
          } else {
            const s = (oy / 2 + 0.5) * (A.py <= B.py ? 1 : -1);
            A.py -= s; B.py += s;
          }
        }
      }
    }
    // 2) weak spring toward the rest point (never all the way, so overlaps that
    //    were just resolved are not immediately undone)
    for (const L of labels) {
      L.px += (L.rx - L.px) * 0.02;
      L.py += (L.ry - L.py) * 0.02;
    }
  }

  return labels.map((L) => {
    const d = toData(L.px, L.py);
    // leader line if the final label is far from where it would sit un-repelled
    const drift = Math.hypot(L.px - L.rx, L.py - L.ry);
    return {
      ...L,
      lx: d.x, ly: d.y,
      tipx: L.x2 || 0, tipy: L.y2 || 0,
      side: L.px >= L.ax ? 'right' : 'left',
      leader: drift > L.h * 0.9
    };
  });
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
