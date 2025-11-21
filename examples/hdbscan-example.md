# HDBSCAN Clustering Example

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a hierarchical extension of DBSCAN that can find clusters of varying density.

## Features

- **Hierarchical clustering**: Builds a hierarchy of clusters
- **Varying density**: Can find clusters of different densities
- **Noise detection**: Identifies outliers as noise (-1 label)
- **Cluster probabilities**: Provides membership probabilities for each point
- **No need to specify number of clusters**: Automatically determines the number of clusters
- **Stable clusters**: Extracts stable clusters from the hierarchy

## Basic Usage

### Class-Based API (Recommended)

```javascript
import { HDBSCAN } from '@tangent.to/ds/ml';

// Create sample data with two clusters
const data = [
  // Cluster 1 (dense)
  [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
  [0.1, 0.1], [0.4, 0.4],
  // Cluster 2 (sparse)
  [10, 10], [11, 10], [10, 11], [11, 11], [10.5, 10.5],
  // Noise points
  [5, 5], [-5, -5]
];

// Create and fit HDBSCAN model
const hdbscan = new HDBSCAN({
  minClusterSize: 5,      // Minimum cluster size
  minSamples: 3,          // Minimum samples for core distance
  clusterSelectionMethod: 'eom'  // Excess of Mass (default)
});

hdbscan.fit(data);

// Get results
console.log('Labels:', hdbscan.labels);
console.log('Probabilities:', hdbscan.probabilities);
console.log('Number of clusters:', hdbscan.nClusters);
console.log('Number of noise points:', hdbscan.nNoise);

// Get summary statistics
const summary = hdbscan.summary();
console.log('Summary:', summary);
/*
{
  minClusterSize: 5,
  minSamples: 3,
  nClusters: 2,
  nNoise: 2,
  nSamples: 14,
  noiseRatio: 0.14,
  avgProbability: 0.86,
  clusterSizes: { 0: 7, 1: 5 }
}
*/

// Predict new points
const newData = [
  [0.3, 0.3],   // Close to cluster 1
  [10.3, 10.3], // Close to cluster 2
  [5, 5]        // Noise
];

const predictions = hdbscan.predict(newData);
console.log('Predicted labels:', predictions.labels);
console.log('Predicted probabilities:', predictions.probabilities);
```

### Functional API

```javascript
import { fit, predict } from '@tangent.to/ds/ml/hdbscan';

const data = [
  [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
  [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
];

// Fit model
const model = fit(data, {
  minClusterSize: 3,
  minSamples: 3
});

console.log('Labels:', model.labels);
console.log('Hierarchy:', model.hierarchy);
console.log('Condensed tree:', model.condensedTree);

// Predict
const newData = [[0.3, 0.3], [10.3, 10.3]];
const predictions = predict(model, newData, data);
```

## Table-Style Input

HDBSCAN supports declarative table-style input:

```javascript
import { HDBSCAN } from '@tangent.to/ds/ml';

const data = [
  { x: 0, y: 0, name: 'A' },
  { x: 0.5, y: 0, name: 'B' },
  { x: 0, y: 0.5, name: 'C' },
  { x: 10, y: 10, name: 'D' },
  { x: 10.5, y: 10, name: 'E' }
];

const hdbscan = new HDBSCAN({ minClusterSize: 2 });
hdbscan.fit({ data, columns: ['x', 'y'] });

console.log('Clusters:', hdbscan.labels);
```

## Visualization

### Condensed Tree Visualization

```javascript
import { HDBSCAN } from '@tangent.to/ds/ml';
import { plotHDBSCAN, plotCondensedTree } from '@tangent.to/ds/plot';
import * as Plot from '@observablehq/plot';

const hdbscan = new HDBSCAN({ minClusterSize: 5 });
hdbscan.fit(data);

// Plot condensed cluster tree
const treeSpec = plotCondensedTree(hdbscan, {
  width: 800,
  height: 600,
  showStability: true
});

// Render (requires custom dendrogram renderer)
const svg = treeSpec.show(customDendrogramRenderer);
```

### Cluster Membership Visualization

```javascript
import { plotClusterMembership } from '@tangent.to/ds/plot';
import * as Plot from '@observablehq/plot';

// Visualize cluster membership with probabilities
const membershipSpec = plotClusterMembership(hdbscan, data, {
  width: 720,
  height: 480,
  showNoise: true,
  columns: ['x', 'y']
});

const svg = membershipSpec.show(Plot);
```

### Cluster Stability Plot

```javascript
import { plotClusterStability } from '@tangent.to/ds/plot';

const stabilitySpec = plotClusterStability(hdbscan, {
  width: 600,
  height: 400
});

const svg = stabilitySpec.show(Plot);
```

### Complete Dashboard

```javascript
import { plotHDBSCANDashboard } from '@tangent.to/ds/plot';

const dashboard = plotHDBSCANDashboard(hdbscan, data, {
  width: 1200,
  height: 800
});

// Access individual plots
dashboard.plots.membership.show(Plot);
dashboard.plots.condensedTree.show(customRenderer);
dashboard.plots.stability.show(Plot);
```

## Accessing Hierarchy Information

```javascript
// Get the full hierarchy
const hierarchy = hdbscan.getHierarchy();
console.log('Dendrogram:', hierarchy.dendrogram);
console.log('Linkage matrix:', hierarchy.linkageMatrix);

// Get condensed tree
const condensedTree = hdbscan.getCondensedTree();
console.log('Condensed tree:', condensedTree);

// Get cluster persistence
const persistence = hdbscan.getClusterPersistence();
console.log('Cluster persistence:', persistence);
```

## Model Persistence

```javascript
// Serialize model
const json = hdbscan.toJSON();
localStorage.setItem('hdbscan-model', JSON.stringify(json));

// Deserialize model
const loadedJson = JSON.parse(localStorage.getItem('hdbscan-model'));
const loadedModel = HDBSCAN.fromJSON(loadedJson);

// Use loaded model
const predictions = loadedModel.predict(newData);
```

## Parameter Tuning

### `minClusterSize`
- Controls the minimum size of clusters
- Larger values = fewer, larger clusters
- Smaller values = more, smaller clusters
- **Recommendation**: Start with `minClusterSize = 5` and adjust based on your data

### `minSamples`
- Controls the conservative-ness of clustering
- Affects core distance calculation
- Defaults to `minClusterSize` if not specified
- **Recommendation**: Use same value as `minClusterSize` or slightly larger for noisier data

### `clusterSelectionMethod`
- `'eom'` (Excess of Mass): Selects clusters based on stability (default)
- `'leaf'`: Selects all leaf clusters
- **Recommendation**: Use `'eom'` for most cases

## Comparison with DBSCAN

| Feature | DBSCAN | HDBSCAN |
|---------|--------|---------|
| Varying density | ❌ No | ✅ Yes |
| Number of parameters | 2 (eps, minSamples) | 2 (minClusterSize, minSamples) |
| Hierarchy | ❌ No | ✅ Yes |
| Cluster probabilities | ❌ No | ✅ Yes |
| Parameter sensitivity | High | Lower |
| Computational complexity | O(n²) | O(n²) |

## Best Practices

1. **Start simple**: Begin with default parameters and visualize results
2. **Scale your data**: HDBSCAN is sensitive to scale - consider standardizing features
3. **Tune minClusterSize**: Based on the minimum meaningful cluster size in your domain
4. **Visualize the hierarchy**: Use the condensed tree to understand cluster structure
5. **Check probabilities**: Low probabilities indicate uncertain cluster membership
6. **Handle noise**: Points with label -1 are noise/outliers

## Limitations & Future Improvements

**Current Implementation**:
- Simplified cluster extraction algorithm
- May not always find optimal clusters in complex scenarios
- Computational complexity is O(n²) for distance matrix

**Planned Improvements**:
- Full cluster stability calculation
- Approximate nearest neighbor search for better performance
- Support for custom distance metrics
- Parallel processing for large datasets

## References

- Campello, R. J., Moulavi, D., & Sander, J. (2013). "Density-based clustering based on hierarchical density estimates"
- McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering"
