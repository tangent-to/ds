/**
 * Upper tail of the studentized range for independent standard normals,
 * divided by an independent sqrt(chi-square(df) / df). Integrate over the
 * minimum normal observation and then over the residual scale.
 */

import { special, studentT } from '@tangent.to/proba';
import { rootScalar } from '../core/optimize.js';

// Positive nodes and weights of the eight-point Gauss–Legendre rule.
const NODES = [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
const WEIGHTS = [0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
const NORMAL_LIMIT = 10;
// Beyond this range, even the pairwise normal tails underflow.
const RANGE_LIMIT = 56;

function integrate(fn, lower, upper, maxWidth = 1) {
  const count = Math.max(1, Math.ceil((upper - lower) / maxWidth));
  const width = (upper - lower) / count;
  const half = width / 2;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const mid = lower + (i + 0.5) * width;
    for (let j = 0; j < NODES.length; j++) {
      const offset = half * NODES[j];
      sum += WEIGHTS[j] * (fn(mid - offset) + fn(mid + offset));
    }
  }
  return half * sum;
}

function normalRangeSF(range, k) {
  if (range === 0) return 1;
  if (range >= RANGE_LIMIT) return 0;
  return k * integrate((z) => {
    const aboveMin = special.normalCdf(-z);
    const aboveRange = special.normalCdf(-z - range);
    // Given the minimum z, at least one of the other k−1 normals must be
    // above z + range. expm1 preserves that difference in the upper tail.
    const exceed = -Math.expm1((k - 1) * Math.log1p(-aboveRange / aboveMin));
    return INV_SQRT_2PI * Math.exp(-z * z / 2) * aboveMin ** (k - 1) * exceed;
  }, -NORMAL_LIMIT - range / 2, NORMAL_LIMIT);
}

// log(1+x) − x, including when df is large and x is close to zero.
function log1pMinusX(x) {
  if (Math.abs(x) < 1e-4) return x * x * (-0.5 + x * (1 / 3 - x / 4));
  return Math.log1p(x) - x;
}

/** Construct a survival function for one combination of group count and df. */
export function studentizedRangeSF(k, df) {
  if (k === 2) {
    // The range of two means is sqrt(2) times the absolute t statistic.
    const params = { nu: df, mu: 0, sigma: 1 };
    return (q) => q <= 0 ? 1 : 2 * studentT.cdf(-q / Math.SQRT2, params);
  }

  // Center the chi density at its mode, then normalize by quadrature. The
  // log-density ratio avoids subtracting large, nearly equal log-gammas.
  const mode = Math.sqrt(df - 1);
  const scale = Math.sqrt(df);
  const lower = Math.max(0, mode - NORMAL_LIMIT);
  const upper = mode + NORMAL_LIMIT;
  const weight = (y) => {
    const x = y - mode;
    let adjustment = 0;
    if (mode !== 0) {
      const relative = x / mode;
      const centeredLog = relative < -0.5
        ? Math.log(y / mode) - relative
        : log1pMinusX(relative);
      adjustment = (df - 1) * centeredLog;
    }
    return Math.exp(adjustment - x * x / 2);
  };
  const normalizer = integrate(weight, lower, upper);

  return (q) => {
    if (q <= 0) return 1;
    const limit = Math.min(upper, scale * RANGE_LIMIT / q);
    if (limit <= lower) return 0;
    // At large q and small df, the tail comes from a narrow band of small
    // scales. Keep the range argument from changing too much in one panel.
    const width = Math.min(1, 4 * scale / q);
    const tail = integrate((y) => weight(y) * normalRangeSF(q * (y / scale), k), lower, limit, width);
    return Math.max(0, Math.min(1, tail / normalizer));
  };
}

/** Invert the same upper tail used for the adjusted p-value. */
export function studentizedRangeCriticalValue(sf, alpha) {
  let lower = 0;
  let upper = 1;
  while (sf(upper) > alpha) {
    lower = upper;
    upper *= 2;
    if (upper === Infinity) return Infinity;
  }
  return rootScalar((q) => sf(q) - alpha, { bracket: [lower, upper] }).x;
}
