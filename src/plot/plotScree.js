import { attachShow } from './show.js';

/**
 * Generate scree plot configuration for PCA/ordination results
 * Shows variance explained by each component
 *
 * @param {Object} result - PCA/LDA/RDA result with varianceExplained
 * @param {Object} options - {width, height, cumulative}
 * @returns {Object} Plot configuration
 */
export function plotScree(result, {
  width = 640,
  height = 300,
  cumulative = false,
} = {}) {
  const { varianceExplained } = result;

  if (!varianceExplained) {
    throw new Error('plotScree requires result.varianceExplained');
  }

  const data = varianceExplained.map((variance, i) => {
    const entry = {
      component: i + 1,
      variance: variance * 100,
      label: `PC${i + 1}`,
    };

    if (cumulative) {
      const cumSum = varianceExplained
        .slice(0, i + 1)
        .reduce((sum, v) => sum + v, 0);
      entry.cumulative = cumSum * 100;
    }

    return entry;
  });

  const config = {
    type: 'scree',
    width,
    height,
    data: { components: data },
    axes: {
      x: { label: 'Principal Component' },
      y: { label: 'Variance Explained (%)' },
    },
    marks: [
      {
        type: 'line',
        data: 'components',
        x: 'component',
        y: cumulative ? 'cumulative' : 'variance',
        stroke: 'steelblue',
        strokeWidth: 2,
      },
      {
        type: 'dot',
        data: 'components',
        x: 'component',
        y: cumulative ? 'cumulative' : 'variance',
        fill: 'steelblue',
        r: 5,
      },
    ],
  };

  return attachShow(config);
}
