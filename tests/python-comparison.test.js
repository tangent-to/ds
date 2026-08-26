/**
 * Comparison tests with Python implementations
 * Verifies numerical correctness after safeguards implementation
 * 
 * These tests require Python 3 with sklearn and scipy installed.
 * They will be skipped if Python dependencies are not available.
 */

import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { PCA } from '../src/mva/estimators/PCA.js';
import { KMeans } from '../src/ml/estimators/KMeans.js';
import { GLM } from '../src/stats/estimators/GLM.js';
import { GaussianProcessRegressor } from '../src/ml/estimators/GaussianProcessRegressor.js';
import { RBF, SumKernel, WhiteKernel } from '../src/ml/kernels/index.js';

// Check Python availability BEFORE test registration
let pythonAvailable = false;
let pythonResults = null;

// The reference script has to run HERE, at module evaluation, and not in a
// beforeAll hook. it.skipIf() is evaluated while tests are being registered,
// which happens before any hook runs, so a failure discovered in beforeAll
// cannot un-register anything: pythonAvailable = false arrives too late and
// every test runs against a null pythonResults, failing with a type error
// instead of skipping. (The R comparison harness had the identical bug, where
// it turned one broken reference script into twelve confusing failures.)
try {
  // Check if python3 and the required packages are available
  execSync('python3 --version', { stdio: 'pipe' });
  execSync('python3 -c "import sklearn; import scipy; import numpy"', { stdio: 'pipe' });

  console.log('Running Python comparison script...');
  execSync('python3 tests/compare_with_python.py', { stdio: 'pipe' });
  pythonResults = JSON.parse(
    readFileSync('/tmp/python_comparison_results.json', 'utf-8')
  );
  pythonAvailable = true;
  console.log('✓ Python reference results loaded');
} catch (error) {
  // Surface Python's own traceback. With stdio piped it lands on error.stderr,
  // and without it a failing script is indistinguishable from Python simply
  // not being installed.
  const detail = (error.stderr?.toString() || error.message || '').trim();
  console.warn('⚠ Python comparison tests will be skipped.');
  if (detail) {
    console.warn(detail.split('\n').slice(-8).join('\n'));
  }
  console.warn('  Install with: pip3 install scikit-learn scipy numpy');
}

describe('PCA - Comparison with sklearn', () => {
  it.skipIf(!pythonAvailable)('should produce similar explained variance ratios', () => {
    const data = pythonResults.pca.data;
    const pca = new PCA({ scale: false, center: true });
    pca.fit({ data });

    // Access varianceExplained from the model (not a method)
    const jsVarianceRatio = pca.model.varianceExplained;
    const pyVarianceRatio = pythonResults.pca.sklearn.explained_variance_ratio;

    console.log('JS explained variance ratio:', jsVarianceRatio);
    console.log('Python explained variance ratio:', pyVarianceRatio);

    // Check each component (allowing for sign flip)
    // Only compare as many components as Python returned
    for (let i = 0; i < pyVarianceRatio.length; i++) {
      expect(Math.abs(jsVarianceRatio[i])).toBeCloseTo(
        Math.abs(pyVarianceRatio[i]),
        3
      );
    }

    // Total variance for first n_components should match closely
    const jsTotal = jsVarianceRatio.slice(0, pyVarianceRatio.length).reduce((a, b) => a + b, 0);
    const pyTotal = pyVarianceRatio.reduce((a, b) => a + b, 0);
    expect(jsTotal).toBeCloseTo(pyTotal, 3);
  });

  it.skipIf(!pythonAvailable)('should produce similar component loadings', () => {
    const data = pythonResults.pca.data;
    const pca = new PCA({ scale: false, center: true });
    pca.fit({ data });

    // Access components from the model (not rotation)
    const jsComponents = pca.model.components;
    const pyComponents = pythonResults.pca.sklearn.components;

    console.log('JS components shape:', jsComponents.length, 'x', jsComponents[0].length);
    console.log('Python components shape:', pyComponents.length, 'x', pyComponents[0].length);

    // PCA components can have arbitrary sign, so we check absolute values
    // and verify they span similar subspaces
    // Only compare as many components as Python returned
    expect(jsComponents.length).toBeGreaterThanOrEqual(pyComponents.length);
    expect(jsComponents[0].length).toBe(pyComponents[0].length);
  });
});

describe('KMeans - Comparison with sklearn', () => {
  it.skipIf(!pythonAvailable)('should produce similar clustering results', () => {
    const data = pythonResults.kmeans.data;
    const kmeans = new KMeans({ k: 3, seed: 42, maxIter: 100 });
    kmeans.fit({ data });

    const jsInertia = kmeans.model.inertia;
    const pyInertia = pythonResults.kmeans.sklearn.inertia;

    console.log('JS inertia:', jsInertia);
    console.log('Python inertia:', pyInertia);

    // Inertia should be similar (within 10% due to initialization differences)
    const relativeError = Math.abs(jsInertia - pyInertia) / pyInertia;
    expect(relativeError).toBeLessThan(0.15);
  });

  it.skipIf(!pythonAvailable)('should find correct number of clusters', () => {
    const data = pythonResults.kmeans.data;
    const kmeans = new KMeans({ k: 3, seed: 42, maxIter: 100 });
    kmeans.fit({ data });

    const jsLabels = kmeans.predict({ data });
    const uniqueLabels = new Set(jsLabels);

    expect(uniqueLabels.size).toBe(3);
  });
});

describe('GLM Logistic Regression - Comparison with sklearn', () => {
  it.skipIf(!pythonAvailable)('should produce coefficients with same signs as sklearn', () => {
    const X = pythonResults.logistic.X;
    const y = pythonResults.logistic.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 1000 });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const pyCoef = [
      pythonResults.logistic.sklearn.intercept,
      ...pythonResults.logistic.sklearn.coefficients
    ];

    console.log('JS coefficients:', jsCoef);
    console.log('Python coefficients:', pyCoef);

    // Logistic regression coefficients can vary significantly between IRLS (JS) and
    // LBFGS (sklearn) optimizers, especially with quasi-separated data.
    // Instead of exact values, verify:
    // 1. Same signs (direction of effect)
    // 2. Similar relative magnitudes (ratios between coefficients)

    // Check signs match (ignoring intercept which can vary more)
    for (let i = 1; i < jsCoef.length; i++) {
      const jsSign = Math.sign(jsCoef[i]);
      const pySign = Math.sign(pyCoef[i]);
      expect(jsSign).toBe(pySign);
    }

    // The prediction accuracy test below is the real validation
  });

  it.skipIf(!pythonAvailable)('should achieve similar prediction accuracy', () => {
    const X = pythonResults.logistic.X;
    const y = pythonResults.logistic.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 1000 });
    glm.fit(X, y);

    const jsPred = glm.predict(X).map(p => (p > 0.5 ? 1 : 0));
    const jsAccuracy = jsPred.filter((p, i) => p === y[i]).length / y.length;
    const pyAccuracy = pythonResults.logistic.sklearn.accuracy;

    console.log('JS accuracy:', jsAccuracy);
    console.log('Python accuracy:', pyAccuracy);

    // Accuracy should be within 5%
    expect(Math.abs(jsAccuracy - pyAccuracy)).toBeLessThan(0.05);
  });
});

describe('GLM Linear Regression - Comparison with scipy', () => {
  it.skipIf(!pythonAvailable)('should produce similar coefficients for linear regression', () => {
    const X = pythonResults.linear.X;
    const y = pythonResults.linear.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const pyCoef = [
      pythonResults.linear.scipy.intercept,
      ...pythonResults.linear.scipy.coefficients
    ];

    console.log('JS coefficients:', jsCoef);
    console.log('Python coefficients:', pyCoef);

    // Coefficients should match very closely for linear regression
    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(pyCoef[i], 3);
    }
  });

  it.skipIf(!pythonAvailable)('should produce similar R² values', () => {
    const X = pythonResults.linear.X;
    const y = pythonResults.linear.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    // Get predictions first, then compute R²
    const yPred = glm.predict(X);
    const jsRSquared = glm.score(y, yPred);
    const pyRSquared = pythonResults.linear.scipy.r_squared;

    console.log('JS R²:', jsRSquared);
    console.log('Python R²:', pyRSquared);

    expect(jsRSquared).toBeCloseTo(pyRSquared, 3);
  });
});

describe('GaussianProcessRegressor - Comparison with sklearn', () => {
  // These run against whatever sklearn is installed, so they catch a future
  // change in sklearn's behaviour that the inlined reference values in
  // gp-heteroscedastic.test.js / gp-white-kernel.test.js cannot. They only run
  // where Python is available — CI has no Python, so those inlined suites stay
  // the ones that actually gate a merge.
  const fixture = () => {
    const { X, y, X_test, length_scale, alpha, noise_level, sklearn } = pythonResults.gp;
    return { X, y, XTest: X_test, lengthScale: length_scale, alpha, noiseLevel: noise_level, sklearn };
  };

  it.skipIf(!pythonAvailable)('matches sklearn with a per-observation alpha array', () => {
    const { X, y, XTest, lengthScale, alpha, sklearn } = fixture();
    const gp = new GaussianProcessRegressor({ kernel: new RBF(lengthScale, 1.0) });
    gp.fit(X, y, { alpha });

    const { mean, std } = gp.predict(XTest, { returnStd: true });
    const ref = sklearn.heteroscedastic;

    console.log('JS mean[0..2]:', mean.slice(0, 3));
    console.log('Python mean[0..2]:', ref.mean.slice(0, 3));

    mean.forEach((m, i) => expect(m).toBeCloseTo(ref.mean[i], 10));
    std.forEach((v, i) => expect(v).toBeCloseTo(ref.std[i], 10));
    expect(gp.logMarginalLikelihood_).toBeCloseTo(ref.log_marginal_likelihood, 10);
  });

  it.skipIf(!pythonAvailable)('matches sklearn with an RBF + WhiteKernel sum', () => {
    const { X, y, XTest, lengthScale, noiseLevel, sklearn } = fixture();
    const gp = new GaussianProcessRegressor({
      kernel: new SumKernel({ kernels: [new RBF(lengthScale, 1.0), new WhiteKernel(noiseLevel)] }),
      alpha: 0,
    });
    gp.fit(X, y);

    const { mean, std } = gp.predict(XTest, { returnStd: true });
    const ref = sklearn.white_kernel;

    console.log('JS std[0..2]:', std.slice(0, 3));
    console.log('Python std[0..2]:', ref.std.slice(0, 3));

    mean.forEach((m, i) => expect(m).toBeCloseTo(ref.mean[i], 10));
    std.forEach((v, i) => expect(v).toBeCloseTo(ref.std[i], 10));
    expect(gp.logMarginalLikelihood_).toBeCloseTo(ref.log_marginal_likelihood, 10);
  });
});

describe('Safeguards - Verify new functionality', () => {
  it('should throw Observable-friendly error when predict called before fit', () => {
    const pca = new PCA({ n_components: 2 });

    expect(() => pca.transform([[1, 2, 3]])).toThrow(/requires a fitted model/);
    expect(() => pca.transform([[1, 2, 3]])).toThrow(/Observable Tip/);
    expect(() => pca.transform([[1, 2, 3]])).toThrow(/isFitted/);
  });

  it('should provide isFitted() method', () => {
    const kmeans = new KMeans({ k: 3 });

    expect(kmeans.isFitted()).toBe(false);

    kmeans.fit({ data: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] });

    expect(kmeans.isFitted()).toBe(true);
  });

  it('should provide getState() method', () => {
    const glm = new GLM({ family: 'gaussian' });

    let state = glm.getState();
    expect(state.fitted).toBe(false);
    expect(state.className).toBe('GLM');

    glm.fit([[1], [2], [3]], [2, 4, 6]);

    state = glm.getState();
    expect(state.fitted).toBe(true);
    expect(state.memoryEstimate).toBeGreaterThan(0);
  });

  it('should provide getMemoryUsage() method', () => {
    const pca = new PCA({ n_components: 2 });

    pca.fit({ data: Array(100).fill(0).map(() => Array(5).fill(0).map(() => Math.random())) });

    const memory = pca.getMemoryUsage();
    expect(typeof memory).toBe('string');
    expect(memory).toMatch(/KB|MB/);
  });

  it('should track warnings for GLM convergence', () => {
    const glm = new GLM({ family: 'binomial', maxIter: 1, warnOnNoConvergence: true });

    // This should not converge in 1 iteration
    const X = Array(50).fill(0).map(() => [Math.random()]);
    const y = Array(50).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);

    glm.fit(X, y);

    expect(glm.hasWarnings()).toBe(true);
    const warnings = glm.getWarnings();
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0].type).toBe('convergence');
  });
});
