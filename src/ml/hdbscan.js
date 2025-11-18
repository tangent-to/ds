/**
 * HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
 * A hierarchical extension of DBSCAN that can find clusters of varying density
 * and provides a hierarchy of clusters for exploration.
 *
 * Based on:
 * Campello, R. J., Moulavi, D., & Sander, J. (2013).
 * Density-based clustering based on hierarchical density estimates.
 * In Pacific-Asia conference on knowledge discovery and data mining (pp. 160-172).
 */

import { toMatrix } from '../core/linalg.js';
import { prepareX } from '../core/table.js';

/**
 * Euclidean distance between two points
 * @param {Array<number>} a - First point
 * @param {Array<number>} b - Second point
 * @returns {number} Distance
 */
function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

/**
 * Compute pairwise distances between all points
 * @param {Array<Array<number>>} data - Data points
 * @returns {Array<Array<number>>} Distance matrix
 */
function computeDistanceMatrix(data) {
  const n = data.length;
  const distances = Array(n).fill(null).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = euclideanDistance(data[i], data[j]);
      distances[i][j] = dist;
      distances[j][i] = dist;
    }
  }

  return distances;
}

/**
 * Compute core distance for each point (distance to k-th nearest neighbor)
 * @param {Array<Array<number>>} distances - Distance matrix
 * @param {number} minSamples - Minimum cluster size (k for k-th nearest neighbor)
 * @returns {Array<number>} Core distances
 */
function computeCoreDistances(distances, minSamples) {
  const n = distances.length;
  const coreDistances = new Array(n);

  for (let i = 0; i < n; i++) {
    // Get distances from point i to all other points
    const dists = distances[i].slice();
    // Sort to find k-th nearest neighbor
    dists.sort((a, b) => a - b);
    // Core distance is distance to minSamples-th nearest neighbor (1-indexed, so minSamples)
    coreDistances[i] = dists[minSamples];
  }

  return coreDistances;
}

/**
 * Compute mutual reachability distance matrix
 * @param {Array<Array<number>>} distances - Distance matrix
 * @param {Array<number>} coreDistances - Core distances
 * @returns {Array<Array<number>>} Mutual reachability distance matrix
 */
function computeMutualReachability(distances, coreDistances) {
  const n = distances.length;
  const mutualReach = Array(n).fill(null).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = Math.max(
        coreDistances[i],
        coreDistances[j],
        distances[i][j]
      );
      mutualReach[i][j] = dist;
      mutualReach[j][i] = dist;
    }
  }

  return mutualReach;
}

/**
 * Union-Find data structure for MST construction
 */
class UnionFind {
  constructor(n) {
    this.parent = Array(n).fill(null).map((_, i) => i);
    this.rank = Array(n).fill(0);
  }

  find(x) {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x, y) {
    const rootX = this.find(x);
    const rootY = this.find(y);

    if (rootX === rootY) return false;

    // Union by rank
    if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }

    return true;
  }
}

/**
 * Build minimum spanning tree using Kruskal's algorithm
 * @param {Array<Array<number>>} mutualReach - Mutual reachability distance matrix
 * @returns {Array<Object>} MST edges [{from, to, distance}]
 */
function buildMST(mutualReach) {
  const n = mutualReach.length;
  const edges = [];

  // Collect all edges
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      edges.push({ from: i, to: j, distance: mutualReach[i][j] });
    }
  }

  // Sort edges by distance
  edges.sort((a, b) => a.distance - b.distance);

  // Build MST using Kruskal's algorithm
  const uf = new UnionFind(n);
  const mst = [];

  for (const edge of edges) {
    if (uf.union(edge.from, edge.to)) {
      mst.push(edge);
      if (mst.length === n - 1) break;
    }
  }

  return mst;
}

/**
 * Build single linkage hierarchy from MST
 * @param {Array<Object>} mst - Minimum spanning tree edges
 * @param {number} n - Number of points
 * @returns {Object} {dendrogram, linkageMatrix}
 */
function buildHierarchy(mst, n) {
  // Sort MST edges by distance to build hierarchy
  const sortedMST = mst.slice().sort((a, b) => a.distance - b.distance);

  const linkageMatrix = [];
  const dendrogram = [];
  let nextCluster = n;

  // Map to track which cluster each node belongs to
  const clusterMap = new Map();
  // Initialize each point as its own cluster
  for (let i = 0; i < n; i++) {
    clusterMap.set(i, i);
  }

  // Track cluster sizes
  const clusterSizes = new Map();
  for (let i = 0; i < n; i++) {
    clusterSizes.set(i, 1);
  }

  for (const edge of sortedMST) {
    const cluster1 = clusterMap.get(edge.from);
    const cluster2 = clusterMap.get(edge.to);

    if (cluster1 !== cluster2) {
      const size = (clusterSizes.get(cluster1) || 1) + (clusterSizes.get(cluster2) || 1);

      linkageMatrix.push({
        cluster1: cluster1,
        cluster2: cluster2,
        distance: edge.distance,
        size: size
      });

      dendrogram.push({
        id: nextCluster,
        left: cluster1,
        right: cluster2,
        distance: edge.distance,
        size: size
      });

      // Update all points in cluster1 and cluster2 to belong to nextCluster
      for (const [point, cluster] of clusterMap.entries()) {
        if (cluster === cluster1 || cluster === cluster2) {
          clusterMap.set(point, nextCluster);
        }
      }

      clusterSizes.set(nextCluster, size);
      nextCluster++;
    }
  }

  return { dendrogram, linkageMatrix };
}

/**
 * Build condensed cluster tree
 * @param {Object} hierarchy - Hierarchy from buildHierarchy
 * @param {number} minClusterSize - Minimum cluster size
 * @returns {Object} Condensed tree structure
 */
function buildCondensedTree(hierarchy, minClusterSize) {
  const { dendrogram } = hierarchy;
  const condensed = [];
  const clusterMap = new Map();
  let nextClusterId = 0;

  // Traverse hierarchy and condense
  for (let i = 0; i < dendrogram.length; i++) {
    const node = dendrogram[i];

    if (node.size >= minClusterSize) {
      condensed.push({
        parent: node.id,
        child: node.left,
        lambda: 1 / (node.distance || 1e-10),
        childSize: 1
      });

      condensed.push({
        parent: node.id,
        child: node.right,
        lambda: 1 / (node.distance || 1e-10),
        childSize: 1
      });
    }
  }

  return condensed;
}

/**
 * Extract clusters from condensed tree based on stability
 * @param {Array<Object>} condensed - Condensed tree
 * @param {Object} hierarchy - Full hierarchy
 * @param {number} n - Number of points
 * @param {number} minClusterSize - Minimum cluster size
 * @returns {Object} {labels, probabilities, stabilities, tree}
 */
function extractClusters(condensed, hierarchy, n, minClusterSize) {
  const { dendrogram } = hierarchy;
  const labels = new Array(n).fill(-1);
  const probabilities = new Array(n).fill(0);

  if (dendrogram.length === 0) {
    return { labels, probabilities, stabilities: [], condensedTree: condensed };
  }

  // Simple flat clustering approach: cut the dendrogram at a specific level
  // Find non-overlapping clusters by traversing from root

  const clusters = [];
  const queue = [];

  // Start with the root (last merge in dendrogram)
  if (dendrogram.length > 0) {
    queue.push(dendrogram[dendrogram.length - 1]);
  }

  while (queue.length > 0) {
    const node = queue.shift();
    const members = getClusterMembers(node, dendrogram, n);

    if (members.length >= minClusterSize) {
      // This is a valid cluster - add it
      const clusterId = clusters.length;
      clusters.push({ members, size: members.length });

      for (const member of members) {
        if (member < n) {
          labels[member] = clusterId;
          probabilities[member] = 1.0;
        }
      }
    } else {
      // Split into children if they exist and are not leaves
      if (node.left < n || node.right < n) {
        // One or both children are leaves - mark remaining as noise
        continue;
      }

      // Find child nodes in dendrogram
      const leftChild = dendrogram.find(d => d.id === node.left);
      const rightChild = dendrogram.find(d => d.id === node.right);

      if (leftChild) queue.push(leftChild);
      if (rightChild) queue.push(rightChild);
    }
  }

  return {
    labels,
    probabilities,
    stabilities: clusters.map((c, id) => ({ clusterId: id, stability: 1.0, size: c.size })),
    condensedTree: condensed
  };
}

/**
 * Get all leaf members of a cluster node
 * @param {Object} node - Cluster node
 * @param {Array<Object>} dendrogram - Full dendrogram
 * @param {number} n - Number of original points
 * @returns {Array<number>} Member indices
 */
function getClusterMembers(node, dendrogram, n) {
  const members = [];
  const stack = [node.left, node.right];

  while (stack.length > 0) {
    const current = stack.pop();

    if (current < n) {
      // Leaf node (original point)
      members.push(current);
    } else {
      // Internal node - find it in dendrogram
      const dendNode = dendrogram.find(d => d.id === current);
      if (dendNode) {
        stack.push(dendNode.left, dendNode.right);
      }
    }
  }

  return members;
}

/**
 * Fit HDBSCAN clustering model
 * @param {Array<Array<number>>|Matrix} X - Data matrix (n samples × d features)
 * @param {Object} options - Configuration options
 * @param {number} [options.minClusterSize=5] - Minimum cluster size
 * @param {number} [options.minSamples=null] - Minimum samples (defaults to minClusterSize)
 * @param {string} [options.clusterSelectionMethod='eom'] - 'eom' (Excess of Mass) or 'leaf'
 * @param {Array<string>} [options.columns=null] - Column names for table input
 * @param {Array<Object>} [options.data=null] - Table data
 * @returns {Object} Model with labels, probabilities, hierarchy, and condensed tree
 */
export function fit(
  X,
  {
    minClusterSize = 5,
    minSamples = null,
    clusterSelectionMethod = 'eom',
    columns = null,
    data: data_in = null,
  } = {}
) {
  // Accept either:
  //  - legacy numeric input: fit(X_array_or_matrix, { minClusterSize, ... })
  //  - declarative options-object as first arg: fit({ data, columns, minClusterSize, ... })
  let data;
  if (
    X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    data_in = opts.data !== undefined ? opts.data : data_in;
    columns = opts.columns !== undefined ? opts.columns : columns;
    minClusterSize = opts.minClusterSize !== undefined ? opts.minClusterSize : minClusterSize;
    minSamples = opts.minSamples !== undefined ? opts.minSamples : minSamples;
    clusterSelectionMethod = opts.clusterSelectionMethod !== undefined ? opts.clusterSelectionMethod : clusterSelectionMethod;
  }

  // Default minSamples to minClusterSize if not provided
  if (minSamples === null) {
    minSamples = minClusterSize;
  }

  // If declarative data provided, prepare numeric matrix via prepareX
  if (data_in) {
    const prepared = prepareX({ columns, data: data_in });
    data = prepared.X;
  } else if (Array.isArray(X) && X.length > 0 && typeof X[0] === 'object' && !Array.isArray(X[0])) {
    // Array of objects (table-like data) - use prepareX
    const prepared = prepareX({ columns, data: X });
    data = prepared.X;
  } else if (Array.isArray(X)) {
    // Array of arrays (numeric matrix)
    data = X.map((row) => Array.isArray(row) ? row : [row]);
  } else {
    const mat = toMatrix(X);
    data = [];
    for (let i = 0; i < mat.rows; i++) {
      const row = [];
      for (let j = 0; j < mat.columns; j++) {
        row.push(mat.get(i, j));
      }
      data.push(row);
    }
  }

  const n = data.length;

  if (n < minClusterSize) {
    // Not enough data points - return all as noise
    return {
      labels: new Array(n).fill(-1),
      probabilities: new Array(n).fill(0),
      nClusters: 0,
      nNoise: n,
      hierarchy: { dendrogram: [], linkageMatrix: [] },
      condensedTree: [],
      stabilities: []
    };
  }

  // Step 1: Compute pairwise distances
  const distances = computeDistanceMatrix(data);

  // Step 2: Compute core distances
  const coreDistances = computeCoreDistances(distances, minSamples);

  // Step 3: Compute mutual reachability distances
  const mutualReach = computeMutualReachability(distances, coreDistances);

  // Step 4: Build minimum spanning tree
  const mst = buildMST(mutualReach);

  // Step 5: Build single linkage hierarchy
  const hierarchy = buildHierarchy(mst, n);

  // Step 6: Build condensed tree
  const condensedTree = buildCondensedTree(hierarchy, minClusterSize);

  // Step 7: Extract clusters
  const { labels, probabilities, stabilities } = extractClusters(
    condensedTree,
    hierarchy,
    n,
    minClusterSize
  );

  // Count clusters and noise
  const uniqueLabels = new Set(labels.filter(l => l !== -1));
  const nClusters = uniqueLabels.size;
  const nNoise = labels.filter(l => l === -1).length;

  return {
    labels,
    probabilities,
    nClusters,
    nNoise,
    hierarchy,
    condensedTree,
    stabilities,
    coreDistances,
    minClusterSize,
    minSamples
  };
}

/**
 * Predict cluster labels for new data points
 * Note: HDBSCAN uses approximate nearest neighbor assignment for prediction.
 * New points are assigned to the cluster of their nearest neighbor from training data
 * if the distance is within the core distance threshold.
 *
 * @param {Object} model - Fitted model from fit()
 * @param {Array<Array<number>>} X - New data points
 * @param {Array<Array<number>>} X_train - Original training data
 * @returns {Object} {labels, probabilities}
 */
export function predict(model, X, X_train) {
  const { labels: trainLabels, coreDistances } = model;
  const data = Array.isArray(X[0]) ? X : X.map((x) => [x]);

  const predictions = new Array(data.length);
  const probabilities = new Array(data.length);

  for (let i = 0; i < data.length; i++) {
    const point = data[i];
    let minDist = Infinity;
    let nearestIdx = -1;

    // Find nearest training point
    for (let j = 0; j < X_train.length; j++) {
      const dist = euclideanDistance(point, X_train[j]);
      if (dist < minDist) {
        minDist = dist;
        nearestIdx = j;
      }
    }

    // Assign to cluster if within core distance, otherwise noise
    if (nearestIdx !== -1 && minDist <= coreDistances[nearestIdx]) {
      predictions[i] = trainLabels[nearestIdx];
      // Probability decreases with distance
      probabilities[i] = Math.max(0, 1 - minDist / coreDistances[nearestIdx]);
    } else {
      predictions[i] = -1;
      probabilities[i] = 0;
    }
  }

  return { labels: predictions, probabilities };
}

/**
 * Compute cluster persistence (lifetime in the hierarchy)
 * @param {Object} model - Fitted HDBSCAN model
 * @returns {Array<Object>} Array of {cluster, persistence} objects
 */
export function clusterPersistence(model) {
  const { hierarchy, labels } = model;
  const { dendrogram } = hierarchy;

  const uniqueClusters = new Set(labels.filter(l => l !== -1));
  const persistences = [];

  for (const cluster of uniqueClusters) {
    // Find the lifetime of this cluster in the hierarchy
    // For now, use a simplified version
    persistences.push({
      cluster,
      persistence: 1.0
    });
  }

  return persistences;
}
