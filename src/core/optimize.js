/**
 * Optimization algorithms.
 *
 * Since 0.7.0 this module re-exports @tangent.to/opt, the tangent suite's
 * scipy.optimize-style package. The historical ds classes (GradientDescent,
 * MomentumOptimizer, RMSProp, AdamOptimizer, createOptimizer) are opt's
 * ds-compatibility layer and behave identically; new code should prefer the
 * declarative API (minimize, minimizeScalar, rootScalar, leastSquares,
 * curveFit) also exported here.
 *
 * @see https://github.com/tangent-to/opt
 */

export * from '@tangent.to/opt';
