/**
 * Biplot arrow scaling.
 *
 * A PCA hands back its two halves on scales that differ by roughly sqrt(n):
 * site scores are normalized to unit column norm (see `src/mva/pca.js`, where
 * raw scores are divided by the singular value), so each is of order 1/sqrt(n),
 * while loadings stay of order 1. Drawing both at face value gives a biplot
 * whose points are a dot at the origin — which is what `loadingFactor: 1` used
 * to do by default.
 */

import { describe, expect, it } from 'vitest';
import { PCA } from '../src/mva/estimators/PCA.js';
import { ordiplot } from '../src/plot/ordiplot.js';

/** 333 rows of four correlated variables — the shape of a penguins PCA. */
function fixture(n = 333) {
  let s = 42;
  const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const randn = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
  const data = [];
  for (let i = 0; i < n; i++) {
    const f = randn();
    data.push({
      var1: 40 + 3 * f + randn(),
      var2: 17 - 2 * f + randn(),
      var3: 200 + 10 * f + 2 * randn(),
      var4: 4200 + 400 * f + 80 * randn(),
    });
  }
  const pca = new PCA({ scale: true });
  pca.fit({ data });
  return {
    scores: pca.getScores('sites'),
    loadings: pca.getScores('loadings'),
    eigenvalues: pca.model.eigenvalues,
    varianceExplained: pca.model.varianceExplained,
  };
}

const radius = (arr, kx, ky) => Math.max(...arr.map((d) => Math.hypot(d[kx] || 0, d[ky] || 0)));
const scoreRadius = (cfg) => radius(cfg.data.scores, 'x', 'y');
const arrowRadius = (cfg) => radius(cfg.data.loadings, 'x2', 'y2');

describe('loading arrow scaling', () => {
  const result = fixture();

  it('fits the arrows to the score cloud by default', () => {
    const cfg = ordiplot(result, { type: 'pca', showLoadings: true });
    const ratio = arrowRadius(cfg) / scoreRadius(cfg);
    // ARROW_HEADROOM is 0.9: the longest arrow reaches 90% of the cloud.
    expect(ratio).toBeCloseTo(0.9, 6);
  });

  it('leaves the points visible, which is the whole point', () => {
    // The failure this guards: at the old default the score cloud was 13x
    // smaller than the arrows and rendered as a dot at the origin.
    const cfg = ordiplot(result, { type: 'pca', showLoadings: true });
    expect(scoreRadius(cfg)).toBeGreaterThan(0.5 * arrowRadius(cfg));
  });

  it('still honours an explicit factor', () => {
    const auto = ordiplot(result, { type: 'pca', showLoadings: true });
    const raw = ordiplot(result, { type: 'pca', showLoadings: true, loadingFactor: 1 });
    expect(arrowRadius(raw)).toBeGreaterThan(arrowRadius(auto) * 5);
    // An explicit factor scales linearly.
    const twice = ordiplot(result, { type: 'pca', showLoadings: true, loadingFactor: 2 });
    expect(arrowRadius(twice)).toBeCloseTo(arrowRadius(raw) * 2, 8);
  });

  it('makes loadingScale inert under auto, and live under an explicit factor', () => {
    // Auto normalizes by the longest vector, which cancels any constant
    // prefactor; loadingScale only bites when a factor is given.
    const a = ordiplot(result, { type: 'pca', showLoadings: true, loadingScale: 3 });
    const b = ordiplot(result, { type: 'pca', showLoadings: true, loadingScale: 50 });
    expect(arrowRadius(a)).toBeCloseTo(arrowRadius(b), 10);

    const c = ordiplot(result, { type: 'pca', showLoadings: true, loadingFactor: 1, loadingScale: 3 });
    const d = ordiplot(result, { type: 'pca', showLoadings: true, loadingFactor: 1, loadingScale: 6 });
    expect(arrowRadius(d)).toBeCloseTo(arrowRadius(c) * 2, 8);
  });

  it('does not divide by zero when every score sits at the origin', () => {
    const degenerate = { ...result, scores: result.scores.map((d) => ({ ...d, pc1: 0, pc2: 0 })) };
    const cfg = ordiplot(degenerate, { type: 'pca', showLoadings: true });
    expect(cfg.data.loadings.every((d) => Number.isFinite(d.x2) && Number.isFinite(d.y2))).toBe(true);
  });
});
