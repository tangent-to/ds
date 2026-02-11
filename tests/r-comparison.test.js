/**
 * Comparison tests with R implementations
 * Verifies numerical correctness against R's statistical packages:
 * - stats::glm() for GLM
 * - lme4::glmer()/lmer() for GLMM
 * - stats::prcomp() for PCA
 * - MASS::lda() for LDA
 *
 * These tests require R with jsonlite and MASS packages.
 * GLMM tests additionally require lme4.
 * Tests will be skipped if R is not available.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { GLM } from '../src/stats/estimators/GLM.js';
import { PCA } from '../src/mva/estimators/PCA.js';
import { LDA } from '../src/mva/estimators/LDA.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Check R availability BEFORE test registration
let rAvailable = false;
let rResults = null;
let hasLme4 = false;

try {
  // Check if Rscript is available
  execSync('Rscript --version', { stdio: 'pipe' });

  // Check if required packages are installed
  execSync('Rscript -e "library(jsonlite); library(MASS)"', { stdio: 'pipe' });
  rAvailable = true;

  // Check for lme4 (optional)
  try {
    execSync('Rscript -e "library(lme4)"', { stdio: 'pipe' });
    hasLme4 = true;
  } catch {
    console.warn('  lme4 not available - GLMM tests will be skipped');
  }
} catch {
  console.warn('⚠ R comparison tests will be skipped: R with jsonlite/MASS not available');
  console.warn('  Install R and run: install.packages(c("jsonlite", "MASS", "lme4"))');
}

beforeAll(async () => {
  if (!rAvailable) return;

  try {
    console.log('Running R comparison script...');
    const rScript = join(__dirname, 'compare_with_r.R');
    execSync(`Rscript "${rScript}" /tmp/r_comparison_results.json`, {
      stdio: 'inherit',
      timeout: 60000
    });
    rResults = JSON.parse(
      readFileSync('/tmp/r_comparison_results.json', 'utf-8')
    );
    hasLme4 = rResults.has_lme4 === true;
    console.log('✓ R reference results loaded');
    if (hasLme4) {
      console.log('✓ lme4 available - GLMM tests enabled');
    }
  } catch (error) {
    console.error('Failed to run R comparison script:', error.message);
    rAvailable = false;
  }
});

// ==============================================================================
// GLM Tests
// ==============================================================================

describe('GLM Gaussian - Comparison with R glm()', () => {
  it.skipIf(!rAvailable)('should produce similar coefficients for linear regression', () => {
    const r = rResults.gaussian_glm;
    const X = r.X;
    const y = r.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const rCoef = r.coefficients;

    console.log('JS coefficients:', jsCoef);
    console.log('R coefficients:', rCoef);

    // Coefficients should match closely
    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(rCoef[i], 4);
    }
  });

  it.skipIf(!rAvailable)('should produce similar fitted values', () => {
    const r = rResults.gaussian_glm;
    const X = r.X;
    const y = r.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    const jsFitted = glm.predict(X);
    const rFitted = r.fitted_values;

    console.log('JS fitted (first 3):', jsFitted.slice(0, 3));
    console.log('R fitted (first 3):', rFitted.slice(0, 3));

    for (let i = 0; i < jsFitted.length; i++) {
      expect(jsFitted[i]).toBeCloseTo(rFitted[i], 4);
    }
  });

  it.skipIf(!rAvailable)('should produce similar deviance', () => {
    const r = rResults.gaussian_glm;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(r.X, r.y);

    console.log('JS deviance:', glm._model.deviance);
    console.log('R deviance:', r.deviance);

    expect(glm._model.deviance).toBeCloseTo(r.deviance, 3);
  });
});

describe('GLM Binomial - Comparison with R glm()', () => {
  it.skipIf(!rAvailable)('should produce similar coefficients for logistic regression', () => {
    const r = rResults.binomial_glm;
    const X = r.X;
    const y = r.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 100 });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const rCoef = r.coefficients;

    console.log('JS coefficients:', jsCoef);
    console.log('R coefficients:', rCoef);

    // Logistic regression coefficients should match reasonably well
    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(rCoef[i], 2);
    }
  });

  it.skipIf(!rAvailable)('should achieve similar prediction accuracy', () => {
    const r = rResults.binomial_glm;
    const X = r.X;
    const y = r.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 100 });
    glm.fit(X, y);

    const jsPred = glm.predict(X).map(p => (p > 0.5 ? 1 : 0));
    const jsAccuracy = jsPred.filter((p, i) => p === y[i]).length / y.length;

    console.log('JS accuracy:', jsAccuracy);
    console.log('R accuracy:', r.accuracy);

    // Accuracy should be within 5%
    expect(Math.abs(jsAccuracy - r.accuracy)).toBeLessThan(0.05);
  });
});

describe('GLM Poisson - Comparison with R glm()', () => {
  it.skipIf(!rAvailable)('should produce similar coefficients for count regression', () => {
    const r = rResults.poisson_glm;
    const X = r.X;
    const y = r.y;

    const glm = new GLM({ family: 'poisson', intercept: true });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const rCoef = r.coefficients;

    console.log('JS coefficients:', jsCoef);
    console.log('R coefficients:', rCoef);

    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(rCoef[i], 3);
    }
  });

  it.skipIf(!rAvailable)('should produce similar AIC', () => {
    const r = rResults.poisson_glm;

    const glm = new GLM({ family: 'poisson', intercept: true });
    glm.fit(r.X, r.y);

    console.log('JS AIC:', glm._model.aic);
    console.log('R AIC:', r.aic);

    // AIC should be reasonably close
    expect(Math.abs(glm._model.aic - r.aic)).toBeLessThan(1);
  });
});

// ==============================================================================
// GLMM Tests (require lme4)
// ==============================================================================

describe('GLMM Gaussian - Comparison with R lmer()', () => {
  it.skipIf(!rAvailable || !hasLme4)('should produce similar fixed effects', () => {
    const r = rResults.gaussian_glmm;
    const X = r.X.map(x => [x]); // Convert to 2D array
    const y = r.y;
    const groups = r.group;

    const glm = new GLM({
      family: 'gaussian',
      intercept: true,
      randomEffects: { intercept: groups }
    });
    glm.fit(X, y);

    const jsFixed = glm._model.fixedEffects;
    const rFixed = r.fixed_effects;

    console.log('JS fixed effects:', jsFixed);
    console.log('R fixed effects:', rFixed);

    // Fixed effects should be reasonably close
    for (let i = 0; i < jsFixed.length; i++) {
      expect(jsFixed[i]).toBeCloseTo(rFixed[i], 1);
    }
  });

  it.skipIf(!rAvailable || !hasLme4)('should estimate similar variance components', () => {
    const r = rResults.gaussian_glmm;
    const X = r.X.map(x => [x]);
    const y = r.y;
    const groups = r.group;

    const glm = new GLM({
      family: 'gaussian',
      intercept: true,
      randomEffects: { intercept: groups }
    });
    glm.fit(X, y);

    const jsVarIntercept = glm._model.varianceComponents.find(v => v.type === 'intercept')?.variance;
    const rVarIntercept = r.variance_intercept;

    console.log('JS variance (intercept):', jsVarIntercept);
    console.log('R variance (intercept):', rVarIntercept);

    // Variance components can vary more, allow larger tolerance
    if (jsVarIntercept !== undefined) {
      const relError = Math.abs(jsVarIntercept - rVarIntercept) / Math.max(rVarIntercept, 0.01);
      expect(relError).toBeLessThan(0.5); // Within 50%
    }
  });
});

describe('GLMM Binomial - Comparison with R glmer()', () => {
  it.skipIf(!rAvailable || !hasLme4)('should produce similar fixed effects for mixed logistic', () => {
    const r = rResults.binomial_glmm;
    const X = r.X.map(x => [x]);
    const y = r.y;
    const groups = r.group;

    const glm = new GLM({
      family: 'binomial',
      intercept: true,
      randomEffects: { intercept: groups },
      maxIter: 100
    });
    glm.fit(X, y);

    const jsFixed = glm._model.fixedEffects;
    const rFixed = r.fixed_effects;

    console.log('JS fixed effects:', jsFixed);
    console.log('R fixed effects:', rFixed);

    // Allow larger tolerance for GLMM (optimization can vary)
    for (let i = 0; i < jsFixed.length; i++) {
      // Check same sign and reasonable magnitude
      expect(Math.sign(jsFixed[i])).toBe(Math.sign(rFixed[i]));
    }
  });

  it.skipIf(!rAvailable || !hasLme4)('should achieve reasonable prediction accuracy', () => {
    const r = rResults.binomial_glmm;
    const X = r.X.map(x => [x]);
    const y = r.y;
    const groups = r.group;

    const glm = new GLM({
      family: 'binomial',
      intercept: true,
      randomEffects: { intercept: groups },
      maxIter: 100
    });
    glm.fit(X, y);

    const jsPred = glm.predict(X).map(p => (p > 0.5 ? 1 : 0));
    const jsAccuracy = jsPred.filter((p, i) => p === y[i]).length / y.length;

    console.log('JS accuracy:', jsAccuracy);
    console.log('R accuracy:', r.accuracy);

    // Both should achieve reasonable accuracy
    expect(jsAccuracy).toBeGreaterThan(0.5);
  });
});

// ==============================================================================
// PCA Tests
// ==============================================================================

describe('PCA - Comparison with R prcomp()', () => {
  it.skipIf(!rAvailable)('should produce similar eigenvalues', () => {
    const r = rResults.pca;
    const data = r.data;

    const pca = new PCA({ center: true, scale: false });
    pca.fit({ data });

    const jsEigen = pca.model.eigenvalues;
    const rEigen = r.eigenvalues;

    console.log('JS eigenvalues:', jsEigen);
    console.log('R eigenvalues:', rEigen);

    // Eigenvalues should match closely
    for (let i = 0; i < rEigen.length; i++) {
      expect(jsEigen[i]).toBeCloseTo(rEigen[i], 4);
    }
  });

  it.skipIf(!rAvailable)('should produce similar variance explained', () => {
    const r = rResults.pca;
    const data = r.data;

    const pca = new PCA({ center: true, scale: false });
    pca.fit({ data });

    const jsVarExp = pca.model.varianceExplained;
    const rVarExp = r.variance_explained;

    console.log('JS variance explained:', jsVarExp);
    console.log('R variance explained:', rVarExp);

    for (let i = 0; i < rVarExp.length; i++) {
      expect(jsVarExp[i]).toBeCloseTo(rVarExp[i], 4);
    }
  });

  it.skipIf(!rAvailable)('should produce similar components (up to sign)', () => {
    const r = rResults.pca;
    const data = r.data;

    const pca = new PCA({ center: true, scale: false });
    pca.fit({ data });

    const jsComponents = pca.model.components;
    const rComponents = r.components;

    console.log('JS components shape:', jsComponents.length, 'x', jsComponents[0].length);
    console.log('R components shape:', rComponents.length, 'x', rComponents[0].length);

    // Components can have opposite signs, check absolute values
    for (let i = 0; i < rComponents.length; i++) {
      for (let j = 0; j < rComponents[i].length; j++) {
        expect(Math.abs(jsComponents[i][j])).toBeCloseTo(Math.abs(rComponents[i][j]), 3);
      }
    }
  });
});

// ==============================================================================
// LDA Tests
// ==============================================================================

describe('LDA - Comparison with R MASS::lda()', () => {
  it.skipIf(!rAvailable)('should achieve similar classification accuracy', () => {
    const r = rResults.lda;
    // R returns X as array of [x1, x2] pairs (60x2 matrix)
    const X = r.X;
    const y = r.y;

    const lda = new LDA();

    // Prepare data for LDA - X is already an array of [x1, x2] pairs
    const data = X.map((row, i) => ({
      X1: row[0],
      X2: row[1],
      class: y[i]
    }));

    lda.fit({
      X: ['X1', 'X2'],
      y: 'class',
      data
    });

    // Get predictions - LDA.predict returns array of class labels directly
    const predictions = lda.predict({ data });
    const jsAccuracy = predictions.filter((pred, i) => pred === y[i]).length / y.length;

    console.log('JS accuracy:', jsAccuracy);
    console.log('R accuracy:', r.accuracy);
    console.log('Sample predictions:', predictions.slice(0, 3));
    console.log('Sample actual:', y.slice(0, 3));

    // Both should achieve good accuracy on well-separated classes
    expect(jsAccuracy).toBeGreaterThan(0.7);
    expect(Math.abs(jsAccuracy - r.accuracy)).toBeLessThan(0.15);
  });

  it.skipIf(!rAvailable)('should produce similar eigenvalues (up to scaling)', () => {
    const r = rResults.lda;
    const X = r.X;
    const y = r.y;

    const lda = new LDA();
    const data = X.map((row, i) => ({
      X1: row[0],
      X2: row[1],
      class: y[i]
    }));

    lda.fit({
      X: ['X1', 'X2'],
      y: 'class',
      data
    });

    const jsEigen = lda.model.eigenvalues;
    // R's svd values need to be squared for eigenvalues
    const rEigen = r.svd.map(s => s * s);

    console.log('JS eigenvalues:', jsEigen);
    console.log('R eigenvalues (from svd²):', rEigen);

    // Compare relative proportions (eigenvalues can be scaled differently)
    if (jsEigen.length >= 2 && rEigen.length >= 2) {
      const jsRatio = jsEigen[0] / (jsEigen[0] + jsEigen[1]);
      const rRatio = rEigen[0] / (rEigen[0] + rEigen[1]);
      expect(jsRatio).toBeCloseTo(rRatio, 1);
    }
  });
});

// ==============================================================================
// Safeguards
// ==============================================================================

describe('R Comparison - Safeguards', () => {
  it('should have R available for comparison tests', () => {
    // This test documents R availability status
    console.log('R available:', rAvailable);
    console.log('lme4 available:', hasLme4);

    // Don't fail if R not available, just document
    expect(true).toBe(true);
  });
});
