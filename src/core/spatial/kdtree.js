/**
 * KD-Tree for efficient nearest neighbor search
 *
 * Reduces nearest neighbor queries from O(n) to O(log n) on average
 * Essential for scaling clustering algorithms to large datasets
 */

/**
 * KD-Tree node
 */
class KDNode {
  constructor(point, index, axis, left = null, right = null) {
    this.point = point;
    this.index = index;
    this.axis = axis;
    this.left = left;
    this.right = right;
  }
}

/**
 * KD-Tree for efficient spatial queries
 */
export class KDTree {
  constructor(points, metric = 'euclidean') {
    this.metric = metric;
    this.dimensions = points[0]?.length || 0;
    this.root = this._build(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  /**
   * Build KD-tree recursively
   * @private
   */
  _build(items, depth) {
    if (items.length === 0) return null;
    if (items.length === 1) {
      return new KDNode(items[0].point, items[0].index, depth % this.dimensions);
    }

    const axis = depth % this.dimensions;

    // Sort by current axis
    items.sort((a, b) => a.point[axis] - b.point[axis]);

    const medianIdx = Math.floor(items.length / 2);
    const median = items[medianIdx];

    return new KDNode(
      median.point,
      median.index,
      axis,
      this._build(items.slice(0, medianIdx), depth + 1),
      this._build(items.slice(medianIdx + 1), depth + 1)
    );
  }

  /**
   * Find k nearest neighbors
   * @param {Array<number>} point - Query point
   * @param {number} k - Number of neighbors
   * @returns {Array<{index: number, distance: number}>}
   */
  knn(point, k) {
    const best = new BoundedPriorityQueue(k);
    this._knnSearch(this.root, point, best);
    return best.toArray().map(item => ({ index: item.index, distance: item.dist }));
  }

  /**
   * Recursive KNN search
   * @private
   */
  _knnSearch(node, point, best) {
    if (!node) return;

    const dist = this._distance(point, node.point);
    best.add({ index: node.index, dist });

    const axis = node.axis;
    const diff = point[axis] - node.point[axis];

    const near = diff < 0 ? node.left : node.right;
    const far = diff < 0 ? node.right : node.left;

    this._knnSearch(near, point, best);

    // Check if we need to search the other side
    if (!best.isFull() || Math.abs(diff) < best.worst()) {
      this._knnSearch(far, point, best);
    }
  }

  /**
   * Find all neighbors within radius
   * @param {Array<number>} point - Query point
   * @param {number} radius - Search radius
   * @returns {Array<{index: number, distance: number}>}
   */
  radiusSearch(point, radius) {
    const neighbors = [];
    this._radiusSearch(this.root, point, radius, neighbors);
    return neighbors;
  }

  /**
   * Recursive radius search
   * @private
   */
  _radiusSearch(node, point, radius, neighbors) {
    if (!node) return;

    const dist = this._distance(point, node.point);
    if (dist <= radius) {
      neighbors.push({ index: node.index, distance: dist });
    }

    const axis = node.axis;
    const diff = point[axis] - node.point[axis];

    const near = diff < 0 ? node.left : node.right;
    const far = diff < 0 ? node.right : node.left;

    this._radiusSearch(near, point, radius, neighbors);

    if (Math.abs(diff) <= radius) {
      this._radiusSearch(far, point, radius, neighbors);
    }
  }

  /**
   * Compute distance between two points
   * @private
   */
  _distance(a, b) {
    switch (this.metric) {
      case 'euclidean':
        return this._euclidean(a, b);
      case 'manhattan':
        return this._manhattan(a, b);
      case 'chebyshev':
        return this._chebyshev(a, b);
      default:
        if (typeof this.metric === 'function') {
          return this.metric(a, b);
        }
        throw new Error(`Unknown metric: ${this.metric}`);
    }
  }

  _euclidean(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  _manhattan(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.abs(a[i] - b[i]);
    }
    return sum;
  }

  _chebyshev(a, b) {
    let max = 0;
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]));
    }
    return max;
  }
}

/**
 * Bounded priority queue for KNN search
 */
class BoundedPriorityQueue {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.items = [];
  }

  add(item) {
    this.items.push(item);
    this.items.sort((a, b) => a.dist - b.dist);
    if (this.items.length > this.maxSize) {
      this.items.pop();
    }
  }

  isFull() {
    return this.items.length >= this.maxSize;
  }

  worst() {
    return this.items[this.items.length - 1]?.dist || Infinity;
  }

  toArray() {
    return this.items;
  }
}

/**
 * Build KD-tree from data
 * @param {Array<Array<number>>} points - Data points
 * @param {string|Function} metric - Distance metric
 * @returns {KDTree}
 */
export function buildKDTree(points, metric = 'euclidean') {
  return new KDTree(points, metric);
}

export default KDTree;
