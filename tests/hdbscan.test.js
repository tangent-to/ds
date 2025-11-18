import { describe, it, expect } from 'vitest';
import { fit as hdbfit, predict as hdbpredict, clusterPersistence } from '../src/ml/hdbscan.js';
import { HDBSCAN } from '../src/ml/index.js';

describe('HDBSCAN clustering', () => {
  describe('functional API', () => {
    it('should cluster simple 2D data with two clusters', () => {
      // Create two well-separated dense clusters
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],  // Cluster 1
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25] // Cluster 2
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.labels.length).toBe(10);
      expect(model.nClusters).toBeGreaterThan(0);
      expect(model.probabilities.length).toBe(10);
      expect(model.hierarchy).toBeDefined();
      expect(model.condensedTree).toBeDefined();
    });

    it('should identify noise points', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],  // Dense cluster
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25], // Dense cluster
        [5, 5], [5.1, 5.1]  // Isolated noise points
      ];

      const model = hdbfit(data, { minClusterSize: 4, minSamples: 3 });

      expect(model.nClusters).toBeGreaterThan(0);
      // Noise points have label -1
      expect(model.labels).toContain(-1);
      expect(model.nNoise).toBeGreaterThan(0);
    });

    it('should handle single cluster', () => {
      const data = [
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [0.5, 0]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.labels.length).toBe(8);
      expect(model.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should handle varying density clusters', () => {
      const data = [
        // Dense cluster (many close points)
        [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [0.15, 0.05], [0.05, 0.15],
        // Sparse cluster (fewer, more spread out points)
        [10, 10], [11, 10], [10, 11], [11, 11], [10.5, 10.5]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.labels.length).toBe(12);
      expect(model.nClusters).toBeGreaterThan(0);
    });

    it('should handle 1D data', () => {
      const data = [
        [1], [2], [3], [4], [10], [11], [12], [13]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 2 });

      expect(model.labels.length).toBe(8);
      expect(model.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should provide cluster probabilities', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.probabilities.length).toBe(10);
      model.probabilities.forEach(prob => {
        expect(prob).toBeGreaterThanOrEqual(0);
        expect(prob).toBeLessThanOrEqual(1);
      });
    });

    it('should predict cluster labels for new points', () => {
      const trainData = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const model = hdbfit(trainData, { minClusterSize: 3, minSamples: 3 });

      const newData = [
        [0.3, 0.3],  // Near cluster 1
        [10.3, 10.3], // Near cluster 2
        [5, 5]        // Far from both (should be noise)
      ];

      const result = hdbpredict(model, newData, trainData);

      expect(result.labels.length).toBe(3);
      expect(result.probabilities.length).toBe(3);
      result.probabilities.forEach(prob => {
        expect(prob).toBeGreaterThanOrEqual(0);
        expect(prob).toBeLessThanOrEqual(1);
      });
    });

    it('should handle hierarchical structure', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.hierarchy).toBeDefined();
      expect(model.hierarchy.dendrogram).toBeDefined();
      expect(model.hierarchy.linkageMatrix).toBeDefined();
      expect(Array.isArray(model.hierarchy.dendrogram)).toBe(true);
    });

    it('should handle small datasets gracefully', () => {
      const data = [
        [0, 0], [1, 1]
      ];

      const model = hdbfit(data, { minClusterSize: 5, minSamples: 3 });

      expect(model.labels.length).toBe(2);
      expect(model.nClusters).toBe(0);
      expect(model.nNoise).toBe(2);
      // All points should be noise
      expect(model.labels.every(l => l === -1)).toBe(true);
    });

    it('should use minSamples parameter correctly', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      // With explicit minSamples
      const model1 = hdbfit(data, { minClusterSize: 3, minSamples: 2 });
      expect(model1.minSamples).toBe(2);

      // Without explicit minSamples (should default to minClusterSize)
      const model2 = hdbfit(data, { minClusterSize: 3 });
      expect(model2.minSamples).toBe(3);
    });
  });

  describe('class-based API (HDBSCAN)', () => {
    it('should fit and predict with class interface', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const hdbscan = new HDBSCAN({ minClusterSize: 3, minSamples: 3 });
      hdbscan.fit(data);

      expect(hdbscan.fitted).toBe(true);
      expect(hdbscan.labels.length).toBe(10);
      expect(hdbscan.probabilities.length).toBe(10);
      expect(hdbscan.nClusters).toBeGreaterThanOrEqual(0);

      const newData = [[0.3, 0.3]];
      const result = hdbscan.predict(newData);
      expect(result.labels.length).toBe(1);
      expect(result.probabilities.length).toBe(1);
    });

    it('should provide hierarchy and condensed tree', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const hdbscan = new HDBSCAN({ minClusterSize: 3 });
      hdbscan.fit(data);

      const hierarchy = hdbscan.getHierarchy();
      expect(hierarchy).toBeDefined();
      expect(hierarchy.dendrogram).toBeDefined();

      const condensedTree = hdbscan.getCondensedTree();
      expect(condensedTree).toBeDefined();
      expect(Array.isArray(condensedTree)).toBe(true);
    });

    it('should provide summary statistics', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25],
        [5, 5]  // Noise point
      ];

      const hdbscan = new HDBSCAN({ minClusterSize: 4, minSamples: 3 });
      hdbscan.fit(data);

      const summary = hdbscan.summary();

      expect(summary.minClusterSize).toBe(4);
      expect(summary.minSamples).toBe(3);
      expect(summary.nClusters).toBeGreaterThanOrEqual(0);
      expect(summary.nSamples).toBe(11);
      expect(summary.noiseRatio).toBeGreaterThanOrEqual(0);
      expect(summary.noiseRatio).toBeLessThanOrEqual(1);
      expect(summary.avgProbability).toBeGreaterThanOrEqual(0);
      expect(summary.avgProbability).toBeLessThanOrEqual(1);
      expect(summary.clusterSizes).toBeDefined();
    });

    it('should throw error when predicting before fitting', () => {
      const hdbscan = new HDBSCAN();
      expect(() => hdbscan.predict([[1, 2]])).toThrow();
    });

    it('should throw error when accessing hierarchy before fitting', () => {
      const hdbscan = new HDBSCAN();
      expect(() => hdbscan.getHierarchy()).toThrow();
      expect(() => hdbscan.getCondensedTree()).toThrow();
    });

    it('should handle declarative table-style input', () => {
      const data = [
        { x: 0, y: 0 },
        { x: 0.5, y: 0 },
        { x: 0, y: 0.5 },
        { x: 0.5, y: 0.5 },
        { x: 0.25, y: 0.25 },
        { x: 10, y: 10 },
        { x: 10.5, y: 10 },
        { x: 10, y: 10.5 },
        { x: 10.5, y: 10.5 },
        { x: 10.25, y: 10.25 }
      ];

      const hdbscan = new HDBSCAN({ minClusterSize: 3 });
      hdbscan.fit({ data, columns: ['x', 'y'] });

      expect(hdbscan.fitted).toBe(true);
      expect(hdbscan.labels.length).toBe(10);
      expect(hdbscan.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should serialize and deserialize', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const hdbscan1 = new HDBSCAN({ minClusterSize: 3, minSamples: 3 });
      hdbscan1.fit(data);

      const json = hdbscan1.toJSON();
      const hdbscan2 = HDBSCAN.fromJSON(json);

      expect(hdbscan2.fitted).toBe(true);
      expect(hdbscan2.labels).toEqual(hdbscan1.labels);
      expect(hdbscan2.probabilities).toEqual(hdbscan1.probabilities);
      expect(hdbscan2.nClusters).toBe(hdbscan1.nClusters);
      expect(hdbscan2.minClusterSize).toBe(3);
      expect(hdbscan2.minSamples).toBe(3);
    });

    it('should support cluster persistence calculation', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const hdbscan = new HDBSCAN({ minClusterSize: 3 });
      hdbscan.fit(data);

      const persistence = hdbscan.getClusterPersistence();
      expect(Array.isArray(persistence)).toBe(true);
    });
  });

  describe('edge cases', () => {
    it('should handle all noise scenario', () => {
      const data = [
        [0, 0], [100, 100], [200, 200], [300, 300]
      ];

      const model = hdbfit(data, { minClusterSize: 5, minSamples: 3 });

      expect(model.nClusters).toBe(0);
      expect(model.nNoise).toBe(4);
      expect(model.labels.every(l => l === -1)).toBe(true);
    });

    it('should handle duplicate points', () => {
      const data = [
        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
        [10, 10], [10, 10], [10, 10], [10, 10], [10, 10]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.labels.length).toBe(10);
      expect(model.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should handle high-dimensional data', () => {
      const data = [
        [0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0], [0.1, 0.1, 0, 0], [0.05, 0.05, 0, 0],
        [10, 10, 10, 10], [10.1, 10, 10, 10], [10, 10.1, 10, 10], [10.1, 10.1, 10, 10], [10.05, 10.05, 10, 10]
      ];

      const model = hdbfit(data, { minClusterSize: 3, minSamples: 3 });

      expect(model.labels.length).toBe(10);
      expect(model.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should handle varying minClusterSize', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10]  // Small cluster
      ];

      // With small minClusterSize
      const model1 = hdbfit(data, { minClusterSize: 2, minSamples: 2 });
      expect(model1.nClusters).toBeGreaterThanOrEqual(0);

      // With larger minClusterSize
      const model2 = hdbfit(data, { minClusterSize: 5, minSamples: 3 });
      expect(model2.labels.length).toBe(7);
    });
  });

  describe('declarative API', () => {
    it('should accept declarative options as first argument', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25]
      ];

      const model = hdbfit({
        data,
        minClusterSize: 3,
        minSamples: 3
      });

      expect(model.labels.length).toBe(10);
      expect(model.nClusters).toBeGreaterThanOrEqual(0);
    });

    it('should handle table data in declarative style', () => {
      const data = [
        { x: 0, y: 0 },
        { x: 0.5, y: 0 },
        { x: 0, y: 0.5 },
        { x: 0.5, y: 0.5 },
        { x: 0.25, y: 0.25 }
      ];

      const model = hdbfit({
        data,
        columns: ['x', 'y'],
        minClusterSize: 2,
        minSamples: 2
      });

      expect(model.labels.length).toBe(5);
    });
  });
});
