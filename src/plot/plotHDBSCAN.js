/**
 * HDBSCAN visualization helpers
 * Condensed tree and cluster membership visualization
 */

import { attachShow, attachTreeShow } from './show.js';

/**
 * Generate condensed cluster tree visualization configuration
 * @param {Object} model - HDBSCAN model or result from hdbscan.fit()
 * @param {Object} options - Visualization options
 * @param {number} [options.width=800] - Plot width
 * @param {number} [options.height=600] - Plot height
 * @param {boolean} [options.showStability=true] - Show cluster stability scores
 * @returns {Object} Observable Plot-compatible configuration
 */
export function plotCondensedTree(model, {
  width = 800,
  height = 600,
  showStability = true
} = {}) {
  const hierarchy = model.hierarchy || model.getHierarchy?.();
  const condensedTree = model.condensedTree || model.getCondensedTree?.();
  const labels = model.labels;

  if (!hierarchy || !hierarchy.dendrogram) {
    throw new Error('plotCondensedTree requires a fitted HDBSCAN model with hierarchy');
  }

  const { dendrogram } = hierarchy;

  // Build tree data structure for visualization
  const nodes = [];
  const links = [];

  // Process dendrogram to create visualization data
  dendrogram.forEach((node, idx) => {
    const lambda = 1 / (node.distance || 1e-10);

    nodes.push({
      id: node.id,
      lambda: lambda,
      size: node.size,
      isLeaf: node.size === 1
    });

    // Create links between parent and children
    links.push({
      parent: node.id,
      child: node.left,
      lambda: lambda,
      size: node.size
    });

    links.push({
      parent: node.id,
      child: node.right,
      lambda: lambda,
      size: node.size
    });
  });

  return attachTreeShow({
    type: 'condensedTree',
    data: {
      nodes,
      links,
      condensedTree
    },
    config: {
      width,
      height,
      showStability,
      orientation: 'vertical'
    }
  });
}

/**
 * Generate dendrogram visualization from HDBSCAN hierarchy
 * Similar to HCA dendrogram but for HDBSCAN
 * @param {Object} model - HDBSCAN model or result from hdbscan.fit()
 * @param {Object} options - Visualization options
 * @returns {Object} Dendrogram configuration
 */
export function plotHDBSCANDendrogram(model, {
  width = 640,
  height = 400,
  orientation = 'vertical'
} = {}) {
  const hierarchy = model.hierarchy || model.getHierarchy?.();

  if (!hierarchy || !hierarchy.dendrogram) {
    throw new Error('plotHDBSCANDendrogram requires a fitted HDBSCAN model with hierarchy');
  }

  const { dendrogram, linkageMatrix } = hierarchy;
  const n = dendrogram.length > 0 ? dendrogram[0].size : 0;

  // Create initial leaf nodes
  const nodes = Array.from({ length: n }, (_, i) => ({
    id: i,
    height: 0,
    isLeaf: true,
    children: null,
    size: 1
  }));

  // Build tree from merges
  const tree = { nodes: [...nodes], merges: [] };

  dendrogram.forEach((merge, idx) => {
    tree.merges.push({
      id: merge.id,
      cluster1: [merge.left],
      cluster2: [merge.right],
      height: merge.distance,
      size: merge.size,
      children: [merge.left, merge.right]
    });
  });

  return attachTreeShow({
    type: 'dendrogram',
    data: tree,
    config: {
      width,
      height,
      linkage: 'hdbscan',
      orientation
    }
  });
}

/**
 * Visualize cluster membership probabilities
 * @param {Object} model - HDBSCAN model or result from hdbscan.fit()
 * @param {Array<Array<number>>} [data] - Original data for scatter plot (optional)
 * @param {Object} options - Visualization options
 * @param {number} [options.width=720] - Plot width
 * @param {number} [options.height=480] - Plot height
 * @param {boolean} [options.showNoise=true] - Show noise points
 * @param {Array<string>} [options.columns=['x', 'y']] - Column names for 2D projection
 * @returns {Object} Observable Plot-compatible configuration
 */
export function plotClusterMembership(model, data = null, {
  width = 720,
  height = 480,
  showNoise = true,
  columns = ['x', 'y']
} = {}) {
  const labels = model.labels;
  const probabilities = model.probabilities;

  if (!labels || !probabilities) {
    throw new Error('plotClusterMembership requires a fitted HDBSCAN model');
  }

  // Build visualization dataset
  const values = labels.map((label, idx) => {
    const row = {
      index: idx,
      cluster: label === -1 ? 'noise' : String(label),
      probability: probabilities[idx],
      isNoise: label === -1
    };

    // Add coordinates if data provided
    if (data && data[idx]) {
      if (Array.isArray(data[idx])) {
        row.x = data[idx][0];
        row.y = data[idx][1];
      } else if (typeof data[idx] === 'object') {
        row.x = data[idx][columns[0]] || data[idx].x;
        row.y = data[idx][columns[1]] || data[idx].y;
      }
    }

    return row;
  });

  // Filter noise if requested
  const filteredValues = showNoise ? values : values.filter(v => !v.isNoise);

  const marks = [];

  // Add scatter plot if coordinates available
  if (filteredValues.length > 0 && filteredValues[0].x !== undefined) {
    marks.push({
      type: 'dot',
      data: 'values',
      x: 'x',
      y: 'y',
      fill: 'cluster',
      opacity: 'probability',
      r: 4,
      tip: true
    });
  } else {
    // Fallback to probability distribution plot
    marks.push({
      type: 'barX',
      data: 'values',
      x: 'probability',
      y: 'index',
      fill: 'cluster',
      tip: true
    });
  }

  const config = {
    type: 'clusterMembership',
    width,
    height,
    data: {
      values: filteredValues
    },
    axes: {
      x: { label: columns[0] || 'x', grid: true },
      y: { label: columns[1] || 'y', grid: true }
    },
    legend: {
      color: {
        label: 'Cluster',
        domain: [...new Set(filteredValues.map(v => v.cluster))]
      }
    },
    marks
  };

  return attachShow(config);
}

/**
 * Visualize cluster stability and persistence
 * @param {Object} model - HDBSCAN model or result from hdbscan.fit()
 * @param {Object} options - Visualization options
 * @param {number} [options.width=600] - Plot width
 * @param {number} [options.height=400] - Plot height
 * @returns {Object} Observable Plot-compatible configuration
 */
export function plotClusterStability(model, {
  width = 600,
  height = 400
} = {}) {
  const stabilities = model.stabilities || [];
  const labels = model.labels;

  if (!stabilities || stabilities.length === 0) {
    throw new Error('plotClusterStability requires a fitted HDBSCAN model with stability scores');
  }

  // Calculate cluster sizes
  const clusterSizes = {};
  labels.forEach(label => {
    if (label !== -1) {
      clusterSizes[label] = (clusterSizes[label] || 0) + 1;
    }
  });

  // Build dataset
  const values = stabilities.map(s => ({
    cluster: String(s.clusterId),
    stability: s.stability,
    size: clusterSizes[s.clusterId] || 0
  }));

  const config = {
    type: 'clusterStability',
    width,
    height,
    data: {
      values
    },
    axes: {
      x: { label: 'Cluster', tickRotate: -45 },
      y: { label: 'Stability', grid: true }
    },
    marks: [
      {
        type: 'barY',
        data: 'values',
        x: 'cluster',
        y: 'stability',
        fill: 'steelblue',
        tip: true
      }
    ]
  };

  return attachShow(config);
}

/**
 * Create a comprehensive HDBSCAN visualization dashboard
 * @param {Object} model - HDBSCAN model
 * @param {Array<Array<number>>} data - Original data
 * @param {Object} options - Visualization options
 * @returns {Object} Dashboard configuration with multiple plots
 */
export function plotHDBSCANDashboard(model, data, options = {}) {
  return {
    type: 'hdbscanDashboard',
    plots: {
      membership: plotClusterMembership(model, data, options),
      condensedTree: plotCondensedTree(model, options),
      stability: plotClusterStability(model, options)
    }
  };
}

// Main export - defaults to condensed tree visualization
export function plotHDBSCAN(model, options = {}) {
  return plotCondensedTree(model, options);
}

export default plotHDBSCAN;
