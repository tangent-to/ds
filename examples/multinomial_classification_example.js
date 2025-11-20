/**
 * Example demonstrating multinomial classification with GLM
 * Shows both one-vs-rest (ovr) and true multinomial approaches
 */

import ds from '../src/index.js';

// Sample data: Iris-like dataset with 3 classes
const data = ds.DataFrame({
  sepal_length: [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 5.0, 5.5, 6.0, 6.7, 5.8, 6.3],
  sepal_width: [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 3.6, 2.6, 2.2, 3.1, 2.7, 2.5],
  petal_length: [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 1.4, 3.5, 4.0, 4.7, 4.1, 5.0],
  petal_width: [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 0.2, 1.0, 1.0, 1.5, 1.0, 1.9],
  species: ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 
            'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica']
});

console.log("=== Multiclass Classification Examples ===\n");
console.log("Dataset:");
console.log(data.head(12));

// ============================================================
// Method 1: One-vs-Rest (OVR) Multiclass
// ============================================================
console.log("\n\n=== Method 1: One-vs-Rest (OVR) ===");
console.log("Uses binary logistic regression for each class vs all others\n");

const model_ovr = new ds.stats.GLM({
  family: 'binomial',
  multiclass: 'ovr'  // <-- This enables one-vs-rest
});

model_ovr.fit({
  data: data,
  X: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
  y: 'species'
});

console.log("Classes detected:", model_ovr._classes);

// Predict
const predictions_ovr = model_ovr.predict({ data: data });
console.log("\nPredictions (OVR):", predictions_ovr.slice(0, 5));

// Predict probabilities
const probs_ovr = model_ovr.predict({ data: data, type: 'prob' });
console.log("\nProbabilities for first observation (OVR):");
console.log(probs_ovr[0]);

// ============================================================
// Method 2: True Multinomial Logistic Regression
// ============================================================
console.log("\n\n=== Method 2: True Multinomial (Joint Optimization) ===");
console.log("Uses softmax and fits K-1 models jointly\n");

// First, create dummy variables for the target
const species_dummies = data.get('species').map(s => {
  return {
    is_versicolor: s === 'versicolor' ? 1 : 0,
    is_virginica: s === 'virginica' ? 1 : 0
    // Reference category: setosa (both dummies = 0)
  };
});

const data_multinomial = data.copy();
data_multinomial.addColumn('is_versicolor', species_dummies.map(d => d.is_versicolor));
data_multinomial.addColumn('is_virginica', species_dummies.map(d => d.is_virginica));

const model_multinomial = new ds.stats.GLM({
  family: 'binomial'
});

model_multinomial.fit({
  data: data_multinomial,
  X: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
  y: ['is_versicolor', 'is_virginica']  // <-- Multiple binary targets triggers multinomial
});

console.log("Multinomial model fitted!");
console.log("Model type:", model_multinomial._isMultinomial ? "True Multinomial" : "Multi-output");

// Predict probabilities
const probs_multinomial = model_multinomial.predict({ 
  data: data_multinomial, 
  type: 'prob' 
});

console.log("\nProbabilities for first observation (Multinomial):");
console.log("P(versicolor):", probs_multinomial[0].is_versicolor);
console.log("P(virginica):", probs_multinomial[0].is_virginica);
console.log("P(setosa):", 1 - probs_multinomial[0].is_versicolor - probs_multinomial[0].is_virginica);

// ============================================================
// Comparison
// ============================================================
console.log("\n\n=== Summary ===");
console.log("\n1. One-vs-Rest (multiclass: 'ovr'):");
console.log("   - Simpler: fits K binary classifiers independently");
console.log("   - Probabilities may not sum to 1.0");
console.log("   - Good for many classes or imbalanced data");
console.log("   - Usage: { family: 'binomial', multiclass: 'ovr' }");

console.log("\n2. True Multinomial:");
console.log("   - Joint optimization using softmax");
console.log("   - Probabilities always sum to 1.0");
console.log("   - Better calibrated probabilities");
console.log("   - Usage: { family: 'binomial' } with multiple binary targets");

console.log("\n\nTo use multinomial with your penguin data:");
console.log("const model = new ds.stats.GLM({ family: 'binomial', multiclass: 'ovr' });");
console.log("model.fit({ data: penguins, X: ['bill_length', 'bill_depth', ...], y: 'species' });");
