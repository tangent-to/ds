/**
 * Example demonstrating Tukey HSD test with named groups
 */

import ds from '../src/index.js';

// Example penguin body mass data
const adelie_var = [3750, 3800, 3250, 3450, 3650, 3625, 4675, 3475, 4250, 3300];
const chinstrap_var = [3700, 3900, 3800, 3400, 3725, 3600, 3750];
const gentoo_var = [4750, 5700, 5400, 4800, 5200, 5400, 5650, 5700, 5900, 6300];

console.log("=== Using Array Input (original behavior) ===\n");

const anova_array = ds.stats.hypothesis.oneWayAnova([adelie_var, chinstrap_var, gentoo_var]);
const tukey_array = ds.stats.hypothesis.tukeyHSD([adelie_var, chinstrap_var, gentoo_var], { 
  alpha: 0.05, 
  anovaResult: anova_array 
});

console.log("Group Names:", tukey_array.groupNames);
console.log("\nComparisons:");
tukey_array.comparisons.forEach(comp => {
  console.log(`  ${comp.groupLabels[0]} vs ${comp.groupLabels[1]}: p = ${comp.pValue.toFixed(6)}, significant = ${comp.significant}`);
});

console.log("\n\n=== Using Object Input (new behavior with named groups) ===\n");

const anova_object = ds.stats.hypothesis.oneWayAnova([adelie_var, chinstrap_var, gentoo_var]);
const tukey_object = ds.stats.hypothesis.tukeyHSD({
  "Adelie": adelie_var,
  "Chinstrap": chinstrap_var,
  "Gentoo": gentoo_var
}, { 
  alpha: 0.05, 
  anovaResult: anova_object 
});

console.log("Group Names:", tukey_object.groupNames);
console.log("\nComparisons:");
tukey_object.comparisons.forEach(comp => {
  console.log(`  ${comp.groupLabels[0]} vs ${comp.groupLabels[1]}: p = ${comp.pValue.toFixed(6)}, significant = ${comp.significant}`);
});

console.log("\n\nFull result with named groups:");
console.log(JSON.stringify(tukey_object, null, 2));
