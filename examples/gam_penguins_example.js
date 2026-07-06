// ---
// title: GAM regression and classification on penguins
// id: ds-gam-penguins
// ---

// %% [markdown]
/*
This notebook takes the Palmer penguins through a full modelling flow with
generalized additive models. A GAM fits a smooth spline of each numeric
predictor instead of a single slope, so it captures curvature while staying
interpretable term by term.

We do two things: predict a continuous outcome (body mass) with
`ds.ml.GAMRegressor`, and predict the three-way species with the multinomial
`ds.ml.GAMClassifier`. Along the way we use the `ds.ml.recipe` API to one-hot
encode `sex` and make a reproducible train/test split.
*/

// %% [javascript]

import ds from 'https://esm.sh/@tangent.to/ds';

// The 333 complete-case Palmer penguins, inlined so the notebook is fully
// reproducible with no network fetch. Three species; four numeric
// measurements plus sex.
const penguins = [
  { species: 'Adelie', bill_length: 39.1, bill_depth: 18.7, flipper_length: 181, body_mass: 3750, sex: 'male' },
  { species: 'Adelie', bill_length: 39.5, bill_depth: 17.4, flipper_length: 186, body_mass: 3800, sex: 'female' },
  { species: 'Adelie', bill_length: 40.3, bill_depth: 18, flipper_length: 195, body_mass: 3250, sex: 'female' },
  { species: 'Adelie', bill_length: 36.7, bill_depth: 19.3, flipper_length: 193, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 39.3, bill_depth: 20.6, flipper_length: 190, body_mass: 3650, sex: 'male' },
  { species: 'Adelie', bill_length: 38.9, bill_depth: 17.8, flipper_length: 181, body_mass: 3625, sex: 'female' },
  { species: 'Adelie', bill_length: 39.2, bill_depth: 19.6, flipper_length: 195, body_mass: 4675, sex: 'male' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 17.6, flipper_length: 182, body_mass: 3200, sex: 'female' },
  { species: 'Adelie', bill_length: 38.6, bill_depth: 21.2, flipper_length: 191, body_mass: 3800, sex: 'male' },
  { species: 'Adelie', bill_length: 34.6, bill_depth: 21.1, flipper_length: 198, body_mass: 4400, sex: 'male' },
  { species: 'Adelie', bill_length: 36.6, bill_depth: 17.8, flipper_length: 185, body_mass: 3700, sex: 'female' },
  { species: 'Adelie', bill_length: 38.7, bill_depth: 19, flipper_length: 195, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 42.5, bill_depth: 20.7, flipper_length: 197, body_mass: 4500, sex: 'male' },
  { species: 'Adelie', bill_length: 34.4, bill_depth: 18.4, flipper_length: 184, body_mass: 3325, sex: 'female' },
  { species: 'Adelie', bill_length: 46, bill_depth: 21.5, flipper_length: 194, body_mass: 4200, sex: 'male' },
  { species: 'Adelie', bill_length: 37.8, bill_depth: 18.3, flipper_length: 174, body_mass: 3400, sex: 'female' },
  { species: 'Adelie', bill_length: 37.7, bill_depth: 18.7, flipper_length: 180, body_mass: 3600, sex: 'male' },
  { species: 'Adelie', bill_length: 35.9, bill_depth: 19.2, flipper_length: 189, body_mass: 3800, sex: 'female' },
  { species: 'Adelie', bill_length: 38.2, bill_depth: 18.1, flipper_length: 185, body_mass: 3950, sex: 'male' },
  { species: 'Adelie', bill_length: 38.8, bill_depth: 17.2, flipper_length: 180, body_mass: 3800, sex: 'male' },
  { species: 'Adelie', bill_length: 35.3, bill_depth: 18.9, flipper_length: 187, body_mass: 3800, sex: 'female' },
  { species: 'Adelie', bill_length: 40.6, bill_depth: 18.6, flipper_length: 183, body_mass: 3550, sex: 'male' },
  { species: 'Adelie', bill_length: 40.5, bill_depth: 17.9, flipper_length: 187, body_mass: 3200, sex: 'female' },
  { species: 'Adelie', bill_length: 37.9, bill_depth: 18.6, flipper_length: 172, body_mass: 3150, sex: 'female' },
  { species: 'Adelie', bill_length: 40.5, bill_depth: 18.9, flipper_length: 180, body_mass: 3950, sex: 'male' },
  { species: 'Adelie', bill_length: 39.5, bill_depth: 16.7, flipper_length: 178, body_mass: 3250, sex: 'female' },
  { species: 'Adelie', bill_length: 37.2, bill_depth: 18.1, flipper_length: 178, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 39.5, bill_depth: 17.8, flipper_length: 188, body_mass: 3300, sex: 'female' },
  { species: 'Adelie', bill_length: 40.9, bill_depth: 18.9, flipper_length: 184, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 36.4, bill_depth: 17, flipper_length: 195, body_mass: 3325, sex: 'female' },
  { species: 'Adelie', bill_length: 39.2, bill_depth: 21.1, flipper_length: 196, body_mass: 4150, sex: 'male' },
  { species: 'Adelie', bill_length: 38.8, bill_depth: 20, flipper_length: 190, body_mass: 3950, sex: 'male' },
  { species: 'Adelie', bill_length: 42.2, bill_depth: 18.5, flipper_length: 180, body_mass: 3550, sex: 'female' },
  { species: 'Adelie', bill_length: 37.6, bill_depth: 19.3, flipper_length: 181, body_mass: 3300, sex: 'female' },
  { species: 'Adelie', bill_length: 39.8, bill_depth: 19.1, flipper_length: 184, body_mass: 4650, sex: 'male' },
  { species: 'Adelie', bill_length: 36.5, bill_depth: 18, flipper_length: 182, body_mass: 3150, sex: 'female' },
  { species: 'Adelie', bill_length: 40.8, bill_depth: 18.4, flipper_length: 195, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 36, bill_depth: 18.5, flipper_length: 186, body_mass: 3100, sex: 'female' },
  { species: 'Adelie', bill_length: 44.1, bill_depth: 19.7, flipper_length: 196, body_mass: 4400, sex: 'male' },
  { species: 'Adelie', bill_length: 37, bill_depth: 16.9, flipper_length: 185, body_mass: 3000, sex: 'female' },
  { species: 'Adelie', bill_length: 39.6, bill_depth: 18.8, flipper_length: 190, body_mass: 4600, sex: 'male' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 19, flipper_length: 182, body_mass: 3425, sex: 'male' },
  { species: 'Adelie', bill_length: 36, bill_depth: 17.9, flipper_length: 190, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 42.3, bill_depth: 21.2, flipper_length: 191, body_mass: 4150, sex: 'male' },
  { species: 'Adelie', bill_length: 39.6, bill_depth: 17.7, flipper_length: 186, body_mass: 3500, sex: 'female' },
  { species: 'Adelie', bill_length: 40.1, bill_depth: 18.9, flipper_length: 188, body_mass: 4300, sex: 'male' },
  { species: 'Adelie', bill_length: 35, bill_depth: 17.9, flipper_length: 190, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 42, bill_depth: 19.5, flipper_length: 200, body_mass: 4050, sex: 'male' },
  { species: 'Adelie', bill_length: 34.5, bill_depth: 18.1, flipper_length: 187, body_mass: 2900, sex: 'female' },
  { species: 'Adelie', bill_length: 41.4, bill_depth: 18.6, flipper_length: 191, body_mass: 3700, sex: 'male' },
  { species: 'Adelie', bill_length: 39, bill_depth: 17.5, flipper_length: 186, body_mass: 3550, sex: 'female' },
  { species: 'Adelie', bill_length: 40.6, bill_depth: 18.8, flipper_length: 193, body_mass: 3800, sex: 'male' },
  { species: 'Adelie', bill_length: 36.5, bill_depth: 16.6, flipper_length: 181, body_mass: 2850, sex: 'female' },
  { species: 'Adelie', bill_length: 37.6, bill_depth: 19.1, flipper_length: 194, body_mass: 3750, sex: 'male' },
  { species: 'Adelie', bill_length: 35.7, bill_depth: 16.9, flipper_length: 185, body_mass: 3150, sex: 'female' },
  { species: 'Adelie', bill_length: 41.3, bill_depth: 21.1, flipper_length: 195, body_mass: 4400, sex: 'male' },
  { species: 'Adelie', bill_length: 37.6, bill_depth: 17, flipper_length: 185, body_mass: 3600, sex: 'female' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 18.2, flipper_length: 192, body_mass: 4050, sex: 'male' },
  { species: 'Adelie', bill_length: 36.4, bill_depth: 17.1, flipper_length: 184, body_mass: 2850, sex: 'female' },
  { species: 'Adelie', bill_length: 41.6, bill_depth: 18, flipper_length: 192, body_mass: 3950, sex: 'male' },
  { species: 'Adelie', bill_length: 35.5, bill_depth: 16.2, flipper_length: 195, body_mass: 3350, sex: 'female' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 19.1, flipper_length: 188, body_mass: 4100, sex: 'male' },
  { species: 'Adelie', bill_length: 35.9, bill_depth: 16.6, flipper_length: 190, body_mass: 3050, sex: 'female' },
  { species: 'Adelie', bill_length: 41.8, bill_depth: 19.4, flipper_length: 198, body_mass: 4450, sex: 'male' },
  { species: 'Adelie', bill_length: 33.5, bill_depth: 19, flipper_length: 190, body_mass: 3600, sex: 'female' },
  { species: 'Adelie', bill_length: 39.7, bill_depth: 18.4, flipper_length: 190, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 39.6, bill_depth: 17.2, flipper_length: 196, body_mass: 3550, sex: 'female' },
  { species: 'Adelie', bill_length: 45.8, bill_depth: 18.9, flipper_length: 197, body_mass: 4150, sex: 'male' },
  { species: 'Adelie', bill_length: 35.5, bill_depth: 17.5, flipper_length: 190, body_mass: 3700, sex: 'female' },
  { species: 'Adelie', bill_length: 42.8, bill_depth: 18.5, flipper_length: 195, body_mass: 4250, sex: 'male' },
  { species: 'Adelie', bill_length: 40.9, bill_depth: 16.8, flipper_length: 191, body_mass: 3700, sex: 'female' },
  { species: 'Adelie', bill_length: 37.2, bill_depth: 19.4, flipper_length: 184, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 36.2, bill_depth: 16.1, flipper_length: 187, body_mass: 3550, sex: 'female' },
  { species: 'Adelie', bill_length: 42.1, bill_depth: 19.1, flipper_length: 195, body_mass: 4000, sex: 'male' },
  { species: 'Adelie', bill_length: 34.6, bill_depth: 17.2, flipper_length: 189, body_mass: 3200, sex: 'female' },
  { species: 'Adelie', bill_length: 42.9, bill_depth: 17.6, flipper_length: 196, body_mass: 4700, sex: 'male' },
  { species: 'Adelie', bill_length: 36.7, bill_depth: 18.8, flipper_length: 187, body_mass: 3800, sex: 'female' },
  { species: 'Adelie', bill_length: 35.1, bill_depth: 19.4, flipper_length: 193, body_mass: 4200, sex: 'male' },
  { species: 'Adelie', bill_length: 37.3, bill_depth: 17.8, flipper_length: 191, body_mass: 3350, sex: 'female' },
  { species: 'Adelie', bill_length: 41.3, bill_depth: 20.3, flipper_length: 194, body_mass: 3550, sex: 'male' },
  { species: 'Adelie', bill_length: 36.3, bill_depth: 19.5, flipper_length: 190, body_mass: 3800, sex: 'male' },
  { species: 'Adelie', bill_length: 36.9, bill_depth: 18.6, flipper_length: 189, body_mass: 3500, sex: 'female' },
  { species: 'Adelie', bill_length: 38.3, bill_depth: 19.2, flipper_length: 189, body_mass: 3950, sex: 'male' },
  { species: 'Adelie', bill_length: 38.9, bill_depth: 18.8, flipper_length: 190, body_mass: 3600, sex: 'female' },
  { species: 'Adelie', bill_length: 35.7, bill_depth: 18, flipper_length: 202, body_mass: 3550, sex: 'female' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 18.1, flipper_length: 205, body_mass: 4300, sex: 'male' },
  { species: 'Adelie', bill_length: 34, bill_depth: 17.1, flipper_length: 185, body_mass: 3400, sex: 'female' },
  { species: 'Adelie', bill_length: 39.6, bill_depth: 18.1, flipper_length: 186, body_mass: 4450, sex: 'male' },
  { species: 'Adelie', bill_length: 36.2, bill_depth: 17.3, flipper_length: 187, body_mass: 3300, sex: 'female' },
  { species: 'Adelie', bill_length: 40.8, bill_depth: 18.9, flipper_length: 208, body_mass: 4300, sex: 'male' },
  { species: 'Adelie', bill_length: 38.1, bill_depth: 18.6, flipper_length: 190, body_mass: 3700, sex: 'female' },
  { species: 'Adelie', bill_length: 40.3, bill_depth: 18.5, flipper_length: 196, body_mass: 4350, sex: 'male' },
  { species: 'Adelie', bill_length: 33.1, bill_depth: 16.1, flipper_length: 178, body_mass: 2900, sex: 'female' },
  { species: 'Adelie', bill_length: 43.2, bill_depth: 18.5, flipper_length: 192, body_mass: 4100, sex: 'male' },
  { species: 'Adelie', bill_length: 35, bill_depth: 17.9, flipper_length: 192, body_mass: 3725, sex: 'female' },
  { species: 'Adelie', bill_length: 41, bill_depth: 20, flipper_length: 203, body_mass: 4725, sex: 'male' },
  { species: 'Adelie', bill_length: 37.7, bill_depth: 16, flipper_length: 183, body_mass: 3075, sex: 'female' },
  { species: 'Adelie', bill_length: 37.8, bill_depth: 20, flipper_length: 190, body_mass: 4250, sex: 'male' },
  { species: 'Adelie', bill_length: 37.9, bill_depth: 18.6, flipper_length: 193, body_mass: 2925, sex: 'female' },
  { species: 'Adelie', bill_length: 39.7, bill_depth: 18.9, flipper_length: 184, body_mass: 3550, sex: 'male' },
  { species: 'Adelie', bill_length: 38.6, bill_depth: 17.2, flipper_length: 199, body_mass: 3750, sex: 'female' },
  { species: 'Adelie', bill_length: 38.2, bill_depth: 20, flipper_length: 190, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 38.1, bill_depth: 17, flipper_length: 181, body_mass: 3175, sex: 'female' },
  { species: 'Adelie', bill_length: 43.2, bill_depth: 19, flipper_length: 197, body_mass: 4775, sex: 'male' },
  { species: 'Adelie', bill_length: 38.1, bill_depth: 16.5, flipper_length: 198, body_mass: 3825, sex: 'female' },
  { species: 'Adelie', bill_length: 45.6, bill_depth: 20.3, flipper_length: 191, body_mass: 4600, sex: 'male' },
  { species: 'Adelie', bill_length: 39.7, bill_depth: 17.7, flipper_length: 193, body_mass: 3200, sex: 'female' },
  { species: 'Adelie', bill_length: 42.2, bill_depth: 19.5, flipper_length: 197, body_mass: 4275, sex: 'male' },
  { species: 'Adelie', bill_length: 39.6, bill_depth: 20.7, flipper_length: 191, body_mass: 3900, sex: 'female' },
  { species: 'Adelie', bill_length: 42.7, bill_depth: 18.3, flipper_length: 196, body_mass: 4075, sex: 'male' },
  { species: 'Adelie', bill_length: 38.6, bill_depth: 17, flipper_length: 188, body_mass: 2900, sex: 'female' },
  { species: 'Adelie', bill_length: 37.3, bill_depth: 20.5, flipper_length: 199, body_mass: 3775, sex: 'male' },
  { species: 'Adelie', bill_length: 35.7, bill_depth: 17, flipper_length: 189, body_mass: 3350, sex: 'female' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 18.6, flipper_length: 189, body_mass: 3325, sex: 'male' },
  { species: 'Adelie', bill_length: 36.2, bill_depth: 17.2, flipper_length: 187, body_mass: 3150, sex: 'female' },
  { species: 'Adelie', bill_length: 37.7, bill_depth: 19.8, flipper_length: 198, body_mass: 3500, sex: 'male' },
  { species: 'Adelie', bill_length: 40.2, bill_depth: 17, flipper_length: 176, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 41.4, bill_depth: 18.5, flipper_length: 202, body_mass: 3875, sex: 'male' },
  { species: 'Adelie', bill_length: 35.2, bill_depth: 15.9, flipper_length: 186, body_mass: 3050, sex: 'female' },
  { species: 'Adelie', bill_length: 40.6, bill_depth: 19, flipper_length: 199, body_mass: 4000, sex: 'male' },
  { species: 'Adelie', bill_length: 38.8, bill_depth: 17.6, flipper_length: 191, body_mass: 3275, sex: 'female' },
  { species: 'Adelie', bill_length: 41.5, bill_depth: 18.3, flipper_length: 195, body_mass: 4300, sex: 'male' },
  { species: 'Adelie', bill_length: 39, bill_depth: 17.1, flipper_length: 191, body_mass: 3050, sex: 'female' },
  { species: 'Adelie', bill_length: 44.1, bill_depth: 18, flipper_length: 210, body_mass: 4000, sex: 'male' },
  { species: 'Adelie', bill_length: 38.5, bill_depth: 17.9, flipper_length: 190, body_mass: 3325, sex: 'female' },
  { species: 'Adelie', bill_length: 43.1, bill_depth: 19.2, flipper_length: 197, body_mass: 3500, sex: 'male' },
  { species: 'Adelie', bill_length: 36.8, bill_depth: 18.5, flipper_length: 193, body_mass: 3500, sex: 'female' },
  { species: 'Adelie', bill_length: 37.5, bill_depth: 18.5, flipper_length: 199, body_mass: 4475, sex: 'male' },
  { species: 'Adelie', bill_length: 38.1, bill_depth: 17.6, flipper_length: 187, body_mass: 3425, sex: 'female' },
  { species: 'Adelie', bill_length: 41.1, bill_depth: 17.5, flipper_length: 190, body_mass: 3900, sex: 'male' },
  { species: 'Adelie', bill_length: 35.6, bill_depth: 17.5, flipper_length: 191, body_mass: 3175, sex: 'female' },
  { species: 'Adelie', bill_length: 40.2, bill_depth: 20.1, flipper_length: 200, body_mass: 3975, sex: 'male' },
  { species: 'Adelie', bill_length: 37, bill_depth: 16.5, flipper_length: 185, body_mass: 3400, sex: 'female' },
  { species: 'Adelie', bill_length: 39.7, bill_depth: 17.9, flipper_length: 193, body_mass: 4250, sex: 'male' },
  { species: 'Adelie', bill_length: 40.2, bill_depth: 17.1, flipper_length: 193, body_mass: 3400, sex: 'female' },
  { species: 'Adelie', bill_length: 40.6, bill_depth: 17.2, flipper_length: 187, body_mass: 3475, sex: 'male' },
  { species: 'Adelie', bill_length: 32.1, bill_depth: 15.5, flipper_length: 188, body_mass: 3050, sex: 'female' },
  { species: 'Adelie', bill_length: 40.7, bill_depth: 17, flipper_length: 190, body_mass: 3725, sex: 'male' },
  { species: 'Adelie', bill_length: 37.3, bill_depth: 16.8, flipper_length: 192, body_mass: 3000, sex: 'female' },
  { species: 'Adelie', bill_length: 39, bill_depth: 18.7, flipper_length: 185, body_mass: 3650, sex: 'male' },
  { species: 'Adelie', bill_length: 39.2, bill_depth: 18.6, flipper_length: 190, body_mass: 4250, sex: 'male' },
  { species: 'Adelie', bill_length: 36.6, bill_depth: 18.4, flipper_length: 184, body_mass: 3475, sex: 'female' },
  { species: 'Adelie', bill_length: 36, bill_depth: 17.8, flipper_length: 195, body_mass: 3450, sex: 'female' },
  { species: 'Adelie', bill_length: 37.8, bill_depth: 18.1, flipper_length: 193, body_mass: 3750, sex: 'male' },
  { species: 'Adelie', bill_length: 36, bill_depth: 17.1, flipper_length: 187, body_mass: 3700, sex: 'female' },
  { species: 'Adelie', bill_length: 41.5, bill_depth: 18.5, flipper_length: 201, body_mass: 4000, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.5, bill_depth: 17.9, flipper_length: 192, body_mass: 3500, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50, bill_depth: 19.5, flipper_length: 196, body_mass: 3900, sex: 'male' },
  { species: 'Chinstrap', bill_length: 51.3, bill_depth: 19.2, flipper_length: 193, body_mass: 3650, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.4, bill_depth: 18.7, flipper_length: 188, body_mass: 3525, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52.7, bill_depth: 19.8, flipper_length: 197, body_mass: 3725, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.2, bill_depth: 17.8, flipper_length: 198, body_mass: 3950, sex: 'female' },
  { species: 'Chinstrap', bill_length: 46.1, bill_depth: 18.2, flipper_length: 178, body_mass: 3250, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51.3, bill_depth: 18.2, flipper_length: 197, body_mass: 3750, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46, bill_depth: 18.9, flipper_length: 195, body_mass: 4150, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51.3, bill_depth: 19.9, flipper_length: 198, body_mass: 3700, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.6, bill_depth: 17.8, flipper_length: 193, body_mass: 3800, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51.7, bill_depth: 20.3, flipper_length: 194, body_mass: 3775, sex: 'male' },
  { species: 'Chinstrap', bill_length: 47, bill_depth: 17.3, flipper_length: 185, body_mass: 3700, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52, bill_depth: 18.1, flipper_length: 201, body_mass: 4050, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.9, bill_depth: 17.1, flipper_length: 190, body_mass: 3575, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.5, bill_depth: 19.6, flipper_length: 201, body_mass: 4050, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.3, bill_depth: 20, flipper_length: 197, body_mass: 3300, sex: 'male' },
  { species: 'Chinstrap', bill_length: 58, bill_depth: 17.8, flipper_length: 181, body_mass: 3700, sex: 'female' },
  { species: 'Chinstrap', bill_length: 46.4, bill_depth: 18.6, flipper_length: 190, body_mass: 3450, sex: 'female' },
  { species: 'Chinstrap', bill_length: 49.2, bill_depth: 18.2, flipper_length: 195, body_mass: 4400, sex: 'male' },
  { species: 'Chinstrap', bill_length: 42.4, bill_depth: 17.3, flipper_length: 181, body_mass: 3600, sex: 'female' },
  { species: 'Chinstrap', bill_length: 48.5, bill_depth: 17.5, flipper_length: 191, body_mass: 3400, sex: 'male' },
  { species: 'Chinstrap', bill_length: 43.2, bill_depth: 16.6, flipper_length: 187, body_mass: 2900, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.6, bill_depth: 19.4, flipper_length: 193, body_mass: 3800, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.7, bill_depth: 17.9, flipper_length: 195, body_mass: 3300, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52, bill_depth: 19, flipper_length: 197, body_mass: 4150, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.5, bill_depth: 18.4, flipper_length: 200, body_mass: 3400, sex: 'female' },
  { species: 'Chinstrap', bill_length: 49.5, bill_depth: 19, flipper_length: 200, body_mass: 3800, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.4, bill_depth: 17.8, flipper_length: 191, body_mass: 3700, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52.8, bill_depth: 20, flipper_length: 205, body_mass: 4550, sex: 'male' },
  { species: 'Chinstrap', bill_length: 40.9, bill_depth: 16.6, flipper_length: 187, body_mass: 3200, sex: 'female' },
  { species: 'Chinstrap', bill_length: 54.2, bill_depth: 20.8, flipper_length: 201, body_mass: 4300, sex: 'male' },
  { species: 'Chinstrap', bill_length: 42.5, bill_depth: 16.7, flipper_length: 187, body_mass: 3350, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51, bill_depth: 18.8, flipper_length: 203, body_mass: 4100, sex: 'male' },
  { species: 'Chinstrap', bill_length: 49.7, bill_depth: 18.6, flipper_length: 195, body_mass: 3600, sex: 'male' },
  { species: 'Chinstrap', bill_length: 47.5, bill_depth: 16.8, flipper_length: 199, body_mass: 3900, sex: 'female' },
  { species: 'Chinstrap', bill_length: 47.6, bill_depth: 18.3, flipper_length: 195, body_mass: 3850, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52, bill_depth: 20.7, flipper_length: 210, body_mass: 4800, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.9, bill_depth: 16.6, flipper_length: 192, body_mass: 2700, sex: 'female' },
  { species: 'Chinstrap', bill_length: 53.5, bill_depth: 19.9, flipper_length: 205, body_mass: 4500, sex: 'male' },
  { species: 'Chinstrap', bill_length: 49, bill_depth: 19.5, flipper_length: 210, body_mass: 3950, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.2, bill_depth: 17.5, flipper_length: 187, body_mass: 3650, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.9, bill_depth: 19.1, flipper_length: 196, body_mass: 3550, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.5, bill_depth: 17, flipper_length: 196, body_mass: 3500, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.9, bill_depth: 17.9, flipper_length: 196, body_mass: 3675, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.8, bill_depth: 18.5, flipper_length: 201, body_mass: 4450, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.1, bill_depth: 17.9, flipper_length: 190, body_mass: 3400, sex: 'female' },
  { species: 'Chinstrap', bill_length: 49, bill_depth: 19.6, flipper_length: 212, body_mass: 4300, sex: 'male' },
  { species: 'Chinstrap', bill_length: 51.5, bill_depth: 18.7, flipper_length: 187, body_mass: 3250, sex: 'male' },
  { species: 'Chinstrap', bill_length: 49.8, bill_depth: 17.3, flipper_length: 198, body_mass: 3675, sex: 'female' },
  { species: 'Chinstrap', bill_length: 48.1, bill_depth: 16.4, flipper_length: 199, body_mass: 3325, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51.4, bill_depth: 19, flipper_length: 201, body_mass: 3950, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.7, bill_depth: 17.3, flipper_length: 193, body_mass: 3600, sex: 'female' },
  { species: 'Chinstrap', bill_length: 50.7, bill_depth: 19.7, flipper_length: 203, body_mass: 4050, sex: 'male' },
  { species: 'Chinstrap', bill_length: 42.5, bill_depth: 17.3, flipper_length: 187, body_mass: 3350, sex: 'female' },
  { species: 'Chinstrap', bill_length: 52.2, bill_depth: 18.8, flipper_length: 197, body_mass: 3450, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.2, bill_depth: 16.6, flipper_length: 191, body_mass: 3250, sex: 'female' },
  { species: 'Chinstrap', bill_length: 49.3, bill_depth: 19.9, flipper_length: 203, body_mass: 4050, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.2, bill_depth: 18.8, flipper_length: 202, body_mass: 3800, sex: 'male' },
  { species: 'Chinstrap', bill_length: 45.6, bill_depth: 19.4, flipper_length: 194, body_mass: 3525, sex: 'female' },
  { species: 'Chinstrap', bill_length: 51.9, bill_depth: 19.5, flipper_length: 206, body_mass: 3950, sex: 'male' },
  { species: 'Chinstrap', bill_length: 46.8, bill_depth: 16.5, flipper_length: 189, body_mass: 3650, sex: 'female' },
  { species: 'Chinstrap', bill_length: 45.7, bill_depth: 17, flipper_length: 195, body_mass: 3650, sex: 'female' },
  { species: 'Chinstrap', bill_length: 55.8, bill_depth: 19.8, flipper_length: 207, body_mass: 4000, sex: 'male' },
  { species: 'Chinstrap', bill_length: 43.5, bill_depth: 18.1, flipper_length: 202, body_mass: 3400, sex: 'female' },
  { species: 'Chinstrap', bill_length: 49.6, bill_depth: 18.2, flipper_length: 193, body_mass: 3775, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.8, bill_depth: 19, flipper_length: 210, body_mass: 4100, sex: 'male' },
  { species: 'Chinstrap', bill_length: 50.2, bill_depth: 18.7, flipper_length: 198, body_mass: 3775, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.1, bill_depth: 13.2, flipper_length: 211, body_mass: 4500, sex: 'female' },
  { species: 'Gentoo', bill_length: 50, bill_depth: 16.3, flipper_length: 230, body_mass: 5700, sex: 'male' },
  { species: 'Gentoo', bill_length: 48.7, bill_depth: 14.1, flipper_length: 210, body_mass: 4450, sex: 'female' },
  { species: 'Gentoo', bill_length: 50, bill_depth: 15.2, flipper_length: 218, body_mass: 5700, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.6, bill_depth: 14.5, flipper_length: 215, body_mass: 5400, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.5, bill_depth: 13.5, flipper_length: 210, body_mass: 4550, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.4, bill_depth: 14.6, flipper_length: 211, body_mass: 4800, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.7, bill_depth: 15.3, flipper_length: 219, body_mass: 5200, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.3, bill_depth: 13.4, flipper_length: 209, body_mass: 4400, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.8, bill_depth: 15.4, flipper_length: 215, body_mass: 5150, sex: 'male' },
  { species: 'Gentoo', bill_length: 40.9, bill_depth: 13.7, flipper_length: 214, body_mass: 4650, sex: 'female' },
  { species: 'Gentoo', bill_length: 49, bill_depth: 16.1, flipper_length: 216, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.5, bill_depth: 13.7, flipper_length: 214, body_mass: 4650, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.4, bill_depth: 14.6, flipper_length: 213, body_mass: 5850, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.8, bill_depth: 14.6, flipper_length: 210, body_mass: 4200, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.3, bill_depth: 15.7, flipper_length: 217, body_mass: 5850, sex: 'male' },
  { species: 'Gentoo', bill_length: 42, bill_depth: 13.5, flipper_length: 210, body_mass: 4150, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.2, bill_depth: 15.2, flipper_length: 221, body_mass: 6300, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.2, bill_depth: 14.5, flipper_length: 209, body_mass: 4800, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.7, bill_depth: 15.1, flipper_length: 222, body_mass: 5350, sex: 'male' },
  { species: 'Gentoo', bill_length: 50.2, bill_depth: 14.3, flipper_length: 218, body_mass: 5700, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.1, bill_depth: 14.5, flipper_length: 215, body_mass: 5000, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.5, bill_depth: 14.5, flipper_length: 213, body_mass: 4400, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.3, bill_depth: 15.8, flipper_length: 215, body_mass: 5050, sex: 'male' },
  { species: 'Gentoo', bill_length: 42.9, bill_depth: 13.1, flipper_length: 215, body_mass: 5000, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.1, bill_depth: 15.1, flipper_length: 215, body_mass: 5100, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.8, bill_depth: 15, flipper_length: 215, body_mass: 5650, sex: 'male' },
  { species: 'Gentoo', bill_length: 48.2, bill_depth: 14.3, flipper_length: 210, body_mass: 4600, sex: 'female' },
  { species: 'Gentoo', bill_length: 50, bill_depth: 15.3, flipper_length: 220, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.3, bill_depth: 15.3, flipper_length: 222, body_mass: 5250, sex: 'male' },
  { species: 'Gentoo', bill_length: 42.8, bill_depth: 14.2, flipper_length: 209, body_mass: 4700, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.1, bill_depth: 14.5, flipper_length: 207, body_mass: 5050, sex: 'female' },
  { species: 'Gentoo', bill_length: 59.6, bill_depth: 17, flipper_length: 230, body_mass: 6050, sex: 'male' },
  { species: 'Gentoo', bill_length: 49.1, bill_depth: 14.8, flipper_length: 220, body_mass: 5150, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.4, bill_depth: 16.3, flipper_length: 220, body_mass: 5400, sex: 'male' },
  { species: 'Gentoo', bill_length: 42.6, bill_depth: 13.7, flipper_length: 213, body_mass: 4950, sex: 'female' },
  { species: 'Gentoo', bill_length: 44.4, bill_depth: 17.3, flipper_length: 219, body_mass: 5250, sex: 'male' },
  { species: 'Gentoo', bill_length: 44, bill_depth: 13.6, flipper_length: 208, body_mass: 4350, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.7, bill_depth: 15.7, flipper_length: 208, body_mass: 5350, sex: 'male' },
  { species: 'Gentoo', bill_length: 42.7, bill_depth: 13.7, flipper_length: 208, body_mass: 3950, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.6, bill_depth: 16, flipper_length: 225, body_mass: 5700, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.3, bill_depth: 13.7, flipper_length: 210, body_mass: 4300, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.6, bill_depth: 15, flipper_length: 216, body_mass: 4750, sex: 'male' },
  { species: 'Gentoo', bill_length: 50.5, bill_depth: 15.9, flipper_length: 222, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.6, bill_depth: 13.9, flipper_length: 217, body_mass: 4900, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.5, bill_depth: 13.9, flipper_length: 210, body_mass: 4200, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.5, bill_depth: 15.9, flipper_length: 225, body_mass: 5400, sex: 'male' },
  { species: 'Gentoo', bill_length: 44.9, bill_depth: 13.3, flipper_length: 213, body_mass: 5100, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.2, bill_depth: 15.8, flipper_length: 215, body_mass: 5300, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.6, bill_depth: 14.2, flipper_length: 210, body_mass: 4850, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.5, bill_depth: 14.1, flipper_length: 220, body_mass: 5300, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.1, bill_depth: 14.4, flipper_length: 210, body_mass: 4400, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.1, bill_depth: 15, flipper_length: 225, body_mass: 5000, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.5, bill_depth: 14.4, flipper_length: 217, body_mass: 4900, sex: 'female' },
  { species: 'Gentoo', bill_length: 45, bill_depth: 15.4, flipper_length: 220, body_mass: 5050, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.8, bill_depth: 13.9, flipper_length: 208, body_mass: 4300, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.5, bill_depth: 15, flipper_length: 220, body_mass: 5000, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.2, bill_depth: 14.5, flipper_length: 208, body_mass: 4450, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.4, bill_depth: 15.3, flipper_length: 224, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.3, bill_depth: 13.8, flipper_length: 208, body_mass: 4200, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.2, bill_depth: 14.9, flipper_length: 221, body_mass: 5300, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.7, bill_depth: 13.9, flipper_length: 214, body_mass: 4400, sex: 'female' },
  { species: 'Gentoo', bill_length: 54.3, bill_depth: 15.7, flipper_length: 231, body_mass: 5650, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.8, bill_depth: 14.2, flipper_length: 219, body_mass: 4700, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.8, bill_depth: 16.8, flipper_length: 230, body_mass: 5700, sex: 'male' },
  { species: 'Gentoo', bill_length: 49.5, bill_depth: 16.2, flipper_length: 229, body_mass: 5800, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.5, bill_depth: 14.2, flipper_length: 220, body_mass: 4700, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.7, bill_depth: 15, flipper_length: 223, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.7, bill_depth: 15, flipper_length: 216, body_mass: 4750, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.4, bill_depth: 15.6, flipper_length: 221, body_mass: 5000, sex: 'male' },
  { species: 'Gentoo', bill_length: 48.2, bill_depth: 15.6, flipper_length: 221, body_mass: 5100, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.5, bill_depth: 14.8, flipper_length: 217, body_mass: 5200, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.4, bill_depth: 15, flipper_length: 216, body_mass: 4700, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.6, bill_depth: 16, flipper_length: 230, body_mass: 5800, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.5, bill_depth: 14.2, flipper_length: 209, body_mass: 4600, sex: 'female' },
  { species: 'Gentoo', bill_length: 51.1, bill_depth: 16.3, flipper_length: 220, body_mass: 6000, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.2, bill_depth: 13.8, flipper_length: 215, body_mass: 4750, sex: 'female' },
  { species: 'Gentoo', bill_length: 45.2, bill_depth: 16.4, flipper_length: 223, body_mass: 5950, sex: 'male' },
  { species: 'Gentoo', bill_length: 49.1, bill_depth: 14.5, flipper_length: 212, body_mass: 4625, sex: 'female' },
  { species: 'Gentoo', bill_length: 52.5, bill_depth: 15.6, flipper_length: 221, body_mass: 5450, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.4, bill_depth: 14.6, flipper_length: 212, body_mass: 4725, sex: 'female' },
  { species: 'Gentoo', bill_length: 50, bill_depth: 15.9, flipper_length: 224, body_mass: 5350, sex: 'male' },
  { species: 'Gentoo', bill_length: 44.9, bill_depth: 13.8, flipper_length: 212, body_mass: 4750, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.8, bill_depth: 17.3, flipper_length: 228, body_mass: 5600, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.4, bill_depth: 14.4, flipper_length: 218, body_mass: 4600, sex: 'female' },
  { species: 'Gentoo', bill_length: 51.3, bill_depth: 14.2, flipper_length: 218, body_mass: 5300, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.5, bill_depth: 14, flipper_length: 212, body_mass: 4875, sex: 'female' },
  { species: 'Gentoo', bill_length: 52.1, bill_depth: 17, flipper_length: 230, body_mass: 5550, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.5, bill_depth: 15, flipper_length: 218, body_mass: 4950, sex: 'female' },
  { species: 'Gentoo', bill_length: 52.2, bill_depth: 17.1, flipper_length: 228, body_mass: 5400, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.5, bill_depth: 14.5, flipper_length: 212, body_mass: 4750, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.5, bill_depth: 16.1, flipper_length: 224, body_mass: 5650, sex: 'male' },
  { species: 'Gentoo', bill_length: 44.5, bill_depth: 14.7, flipper_length: 214, body_mass: 4850, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.8, bill_depth: 15.7, flipper_length: 226, body_mass: 5200, sex: 'male' },
  { species: 'Gentoo', bill_length: 49.4, bill_depth: 15.8, flipper_length: 216, body_mass: 4925, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.9, bill_depth: 14.6, flipper_length: 222, body_mass: 4875, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.4, bill_depth: 14.4, flipper_length: 203, body_mass: 4625, sex: 'female' },
  { species: 'Gentoo', bill_length: 51.1, bill_depth: 16.5, flipper_length: 225, body_mass: 5250, sex: 'male' },
  { species: 'Gentoo', bill_length: 48.5, bill_depth: 15, flipper_length: 219, body_mass: 4850, sex: 'female' },
  { species: 'Gentoo', bill_length: 55.9, bill_depth: 17, flipper_length: 228, body_mass: 5600, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.2, bill_depth: 15.5, flipper_length: 215, body_mass: 4975, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.1, bill_depth: 15, flipper_length: 228, body_mass: 5500, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.8, bill_depth: 16.1, flipper_length: 215, body_mass: 5500, sex: 'male' },
  { species: 'Gentoo', bill_length: 41.7, bill_depth: 14.7, flipper_length: 210, body_mass: 4700, sex: 'female' },
  { species: 'Gentoo', bill_length: 53.4, bill_depth: 15.8, flipper_length: 219, body_mass: 5500, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.3, bill_depth: 14, flipper_length: 208, body_mass: 4575, sex: 'female' },
  { species: 'Gentoo', bill_length: 48.1, bill_depth: 15.1, flipper_length: 209, body_mass: 5500, sex: 'male' },
  { species: 'Gentoo', bill_length: 50.5, bill_depth: 15.2, flipper_length: 216, body_mass: 5000, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.8, bill_depth: 15.9, flipper_length: 229, body_mass: 5950, sex: 'male' },
  { species: 'Gentoo', bill_length: 43.5, bill_depth: 15.2, flipper_length: 213, body_mass: 4650, sex: 'female' },
  { species: 'Gentoo', bill_length: 51.5, bill_depth: 16.3, flipper_length: 230, body_mass: 5500, sex: 'male' },
  { species: 'Gentoo', bill_length: 46.2, bill_depth: 14.1, flipper_length: 217, body_mass: 4375, sex: 'female' },
  { species: 'Gentoo', bill_length: 55.1, bill_depth: 16, flipper_length: 230, body_mass: 5850, sex: 'male' },
  { species: 'Gentoo', bill_length: 48.8, bill_depth: 16.2, flipper_length: 222, body_mass: 6000, sex: 'male' },
  { species: 'Gentoo', bill_length: 47.2, bill_depth: 13.7, flipper_length: 214, body_mass: 4925, sex: 'female' },
  { species: 'Gentoo', bill_length: 46.8, bill_depth: 14.3, flipper_length: 215, body_mass: 4850, sex: 'female' },
  { species: 'Gentoo', bill_length: 50.4, bill_depth: 15.7, flipper_length: 222, body_mass: 5750, sex: 'male' },
  { species: 'Gentoo', bill_length: 45.2, bill_depth: 14.8, flipper_length: 212, body_mass: 5200, sex: 'female' },
  { species: 'Gentoo', bill_length: 49.9, bill_depth: 16.1, flipper_length: 213, body_mass: 5400, sex: 'male' },
];
const speciesCounts = penguins.reduce((acc, p) => {
  acc[p.species] = (acc[p.species] || 0) + 1;
  return acc;
}, {});

({
  rows: penguins.length,
  columns: Object.keys(penguins[0]),
  speciesCounts, // Adelie 146, Chinstrap 68, Gentoo 119
});

// %% [markdown]
/*
## Preprocessing with a recipe

`ds.ml.recipe` describes preprocessing declaratively and keeps it inspectable.
Here we one-hot encode `sex` (a two-level factor, so it becomes a single
`sex_female` indicator) and split 70/30 with a fixed seed. `prep()` runs the
steps and returns `train` and `test` bundles, each carrying its data, the
resolved feature list `X`, and the target `y`.
*/

// %% [javascript]

const regPrep = ds.ml
  .recipe({
    data: penguins,
    X: ['bill_length', 'bill_depth', 'flipper_length', 'sex'],
    y: 'body_mass',
  })
  .oneHot(['sex'])
  .split({ ratio: 0.7, shuffle: true, seed: 42 })
  .prep();

({
  trainSamples: regPrep.train.data.length, // 233
  testSamples: regPrep.test.data.length,   // 100
  features: regPrep.train.X,               // sex expanded to sex_female
});

// %% [markdown]
/*
## Fitting the body-mass regressor

We fit a GAM with eight-knot cubic regression splines and let GCV choose each
smooth's penalty over a custom grid. The summary reports R-squared, the
residual standard error (in grams), and the effective degrees of freedom
(EDF) — a GAM's answer to "how many parameters did I really use", summed over
the smooths. Each smooth term also gets its own EDF and an approximate
p-value.
*/

// %% [javascript]

const gamReg = new ds.ml.GAMRegressor({
  nSplines: 8,
  basis: 'cr',
  smoothMethod: 'GCV',
  lambdaMin: 1e-6,
  lambdaMax: 1e3,
  nSteps: 25,
});

gamReg.fit({ data: regPrep.train.data, X: regPrep.train.X, y: regPrep.train.y });

const regSummary = gamReg.summary();

({
  call: regSummary.call,
  rSquared: regSummary.rSquared,               // ~0.77
  residualStdError: regSummary.residualStdError, // ~401 g
  edf: regSummary.edf,                          // ~8.75
  n: regSummary.n,                              // 233
  smoothTerms: regSummary.smoothTerms.map((t) => ({
    term: t.term,
    edf: t.edf,
    pValue: t.pValue,
  })),
});

// %% [markdown]
/*
## Scoring the regressor on held-out data

The real test is the 30% we held back. We predict body mass for the test rows
and score with `ds.ml.metrics`: R-squared and RMSE (root mean squared error,
in grams). The test R-squared is lower than training, as expected, but the
model still explains most of the variation in mass.
*/

// %% [javascript]

const regTestPred = gamReg.predict({ data: regPrep.test.data, X: regPrep.test.X });
const regTestActual = regPrep.test.data.map((row) => row.body_mass);

({
  testR2: ds.ml.metrics.r2(regTestActual, regTestPred),                 // ~0.59
  testRMSE: Math.sqrt(ds.ml.metrics.mse(regTestActual, regTestPred)),   // ~496 g
});

// %% [markdown]
/*
Predicted versus actual body mass for the held-out penguins: the closer the
points hug the dashed y = x line, the better the fit. The cloud is tight but
spread wider for the heavier Gentoo.
*/

// %% [javascript]

const regPvA = regPrep.test.data.map((row, i) => ({
  actual: row.body_mass,
  predicted: regTestPred[i],
  species: row.species,
}));
const massLo = d3.min(regPvA, (d) => Math.min(d.actual, d.predicted));
const massHi = d3.max(regPvA, (d) => Math.max(d.actual, d.predicted));
const plot_regPvA = Plot.plot({
  grid: true,
  aspectRatio: 1,
  x: { label: 'Actual body mass (g)' },
  y: { label: 'Predicted body mass (g)' },
  color: { legend: true },
  marks: [
    Plot.line([[massLo, massLo], [massHi, massHi]], { stroke: '#888', strokeDasharray: '4 4' }),
    Plot.dot(regPvA, { x: 'actual', y: 'predicted', stroke: 'species', r: 4 }),
  ],
});
plot_regPvA;

// %% [markdown]
/*
## Classifying species

Now a three-class problem: predict species from the four measurements plus
sex. The recipe encodes the categorical target to integers, so we pass its
`encoders` to the classifier — that lets predictions come back as species
names rather than 0/1/2. `nCoefficients` is K-1 = 2, the multinomial's
reference-coded coefficient vectors.
*/

// %% [javascript]

const clsPrep = ds.ml
  .recipe({
    data: penguins,
    X: ['bill_length', 'bill_depth', 'flipper_length', 'body_mass', 'sex'],
    y: 'species',
  })
  .oneHot(['sex'])
  .split({ ratio: 0.7, shuffle: true, seed: 42 })
  .prep();

const gamCls = new ds.ml.GAMClassifier({ nSplines: 6, basis: 'cr', lambda: 0.1 });
gamCls.fit({
  data: clsPrep.train.data,
  X: clsPrep.train.X,
  y: clsPrep.train.y,
  encoders: clsPrep.train.metadata.encoders,
});

const clsSummary = gamCls.summary();

({
  family: clsSummary.family,                     // multinomial
  link: clsSummary.link,                         // softmax
  classes: clsSummary.classes,                   // [Adelie, Chinstrap, Gentoo]
  nCoefficients: clsSummary.nCoefficients,       // 2
  trainingAccuracy: clsSummary.trainingAccuracy, // ~0.906
  perClassAccuracy: clsSummary.perClassAccuracy,
});

// %% [markdown]
/*
## Test accuracy and confusion matrix

We predict species on the held-out rows, score overall accuracy, and build a
confusion matrix (rows = actual species, columns = predicted). About 90% of
test penguins are classified correctly, with the few errors concentrated where
species overlap.
*/

// %% [javascript]

const clsTestPred = gamCls.predict({ data: clsPrep.test.data, X: clsPrep.test.X });
const clsTestActual = clsPrep.test.data.map((row) => row.species);

const confusion = {};
for (const actual of clsSummary.classes) {
  confusion[actual] = {};
  for (const predicted of clsSummary.classes) confusion[actual][predicted] = 0;
}
for (let i = 0; i < clsTestActual.length; i++) {
  confusion[clsTestActual[i]][clsTestPred[i]]++;
}

({
  testAccuracy: ds.ml.metrics.accuracy(clsTestActual, clsTestPred), // ~0.90
  confusionMatrix: confusion,
});

// %% [markdown]
/*
The confusion matrix as a heatmap (rows = actual, columns = predicted): a
strong diagonal means most penguins are classified correctly, and the few
off-diagonal cells show where species were confused.
*/

// %% [javascript]

const plot_confusion = ds.plot
  .plotConfusionMatrix(clsTestActual, clsTestPred)
  .show(Plot);
plot_confusion;

// %% [markdown]
/*
## Per-row probabilities

`predictProba` returns a probability object per row keyed by species name.
We inspect the first three test penguins alongside their true labels: the
model is confident and correct on all three, which is typical for the clearly
separated Gentoo and for well-measured birds.
*/

// %% [javascript]

const sampleRows = clsPrep.test.data.slice(0, 3);
const samplePreds = gamCls.predict({ data: sampleRows, X: clsPrep.test.X });
const sampleProba = gamCls.predictProba({ data: sampleRows, X: clsPrep.test.X });

sampleRows.map((row, i) => ({
  billLength: row.bill_length,
  flipperLength: row.flipper_length,
  actual: row.species,
  predicted: samplePreds[i],
  probabilities: sampleProba[i],
}));

// %% [markdown]
/*
The predicted species probabilities for those same three penguins: the model
is confident and correct on all three, putting almost all of its mass on the
true species.
*/

// %% [javascript]

const sampleLong = sampleProba.flatMap((row, i) =>
  Object.entries(row).map(([cls, p]) => ({
    penguin: `penguin ${i} (actual ${sampleRows[i].species})`,
    species: cls,
    prob: p,
  })),
);
const plot_sampleProba = Plot.plot({
  marginLeft: 170,
  x: { label: 'Probability', domain: [0, 1] },
  y: { label: 'Species' },
  fy: { label: null },
  color: { legend: true },
  marks: [
    Plot.barX(sampleLong, { x: 'prob', y: 'species', fy: 'penguin', fill: 'species' }),
    Plot.ruleX([0]),
  ],
});
plot_sampleProba;
