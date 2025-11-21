/**
 * Visualization module
 * Observable Plot configuration generators for data analysis
 */

// Unified ordination plot (use this for PCA, LDA, RDA)
export { ordiplot } from './ordiplot.js';

// Scree plot for variance explained
export { plotScree } from './plotScree.js';

// Clustering/hierarchical plots
export { dendrogramLayout, plotHCA } from './plotHCA.js';
export {
  plotHDBSCAN,
  plotCondensedTree,
  plotHDBSCANDendrogram,
  plotClusterMembership,
  plotClusterStability,
  plotHDBSCANDashboard
} from './plotHDBSCAN.js';

// Classification metrics plots
export {
  plotCalibration,
  plotConfusionMatrix,
  plotPrecisionRecall,
  plotROC,
} from './classification.js';

// Model interpretation plots
export {
  plotCorrelationMatrix,
  plotFeatureImportance,
  plotLearningCurve,
  plotPartialDependence,
  plotQQ,
  plotResiduals,
} from './utils.js';

// Renderer helpers
export { createD3DendrogramRenderer } from './renderers/d3Dendrogram.js';

export { plotSilhouette } from './plotSilhouette.js';

// GLM/GLMM diagnostic plots
export {
  diagnosticDashboard,
  effectPlot,
  partialResidualPlot,
  qqPlot,
  residualPlot,
  residualsLeveragePlot,
  scaleLocationPlot,
} from './diagnostics.js';
