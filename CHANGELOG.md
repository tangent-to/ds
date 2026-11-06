# Changelog

Notable changes to `@tangent.to/ds`. This file starts at 0.11.0; for earlier
releases see the [git history](https://github.com/tangent-to/ds/commits/main)
and the [release tags](https://github.com/tangent-to/ds/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.16.0] - 2026-11-06

### Fixed

- **`mva.composition.multiplicativeReplacement` now does a multiplicative
  replacement. Results change for every composition that contains a zero**,
  and so do `clr`, `alr` and `ilr` called with `handleZeros: true`.

  The old code subtracted one amount from every non-zero part of a row:

  ```
  l   = delta · nZeros / (D − nZeros)
  r_j = delta     if x_j = 0
  r_j = x_j − l   otherwise          (then closure)
  ```

  The function now follows Martín-Fernández, Barceló-Vidal & Pawlowsky-Glahn
  (2003, *Mathematical Geology* 35(3), 253–278). It first closes the row, so
  `delta` is a proportion of the row total, then:

  ```
  r_j = δ_j                          if x_j = 0
  r_j = x_j · (1 − Σ_{k : x_k = 0} δ_k)   otherwise
  ```

  The old formula had three defects. (1) A small part went negative as soon as
  it was below `l`, and `closure` then threw "mat cannot contain negative
  values". With seven parties and `delta = (1/7)²`, the row
  `[0, 0, 0.45, 0.038, 0.47, 0.008, 0.033]` failed this way. (2) Ratios
  between non-zero parts were not kept, and keeping them (subcompositional
  coherence) is what justifies the method. (3) It worked on the raw row and
  assumed a total of 1, so a rounded poll summing to 0.98 or 1.03 got
  replacements on the wrong scale.

  `delta` can now also be an array with one threshold per part. The signature
  is unchanged, and there is no option to restore the old formula. The
  function now throws clear errors for a negative value, a row with a zero
  total, a `delta` array of the wrong length or a non-positive `delta`, and a
  row whose zeros' thresholds sum to 1 or more. The zero-total and threshold
  errors name the row.

## [0.15.0] - 2026-09-22

### Added

- **`MLPRegressor` on `@tangent.to/nn`.** The same estimator, its `layerSizes`,
  `activation`, `learningRate`, `epochs`, `batchSize` and `verbose` unchanged,
  now a chain of nn layers differentiated by grad and trained with opt's
  update rules or L-BFGS (`optimizer`). It gains `dropout`, `seed`,
  `normalizeY` (on by default), `predict(X, { samples })` for Monte Carlo
  dropout and `predictGradient(x)`; `summary().network` is nn's. Branches,
  other losses and learned dropout are nn's own API.

### Changed

- `MLPRegressor.predict` returns a flat array for one output, as the other
  regressors do, and rows only for several. `layerSizes` is checked against
  the data: its first entry must be the feature count, its last the number of
  targets.
- `ml.loss` functions return the loss alone, a number, no longer
  `{ loss, gradient }`: the gradients existed for `ml.train`, which is gone.
- `RandomForest` draws its seeded randomness from proba's generator.
- Dependencies: grad ^0.3.1, opt ^0.2.0, proba ^0.2.1, and nn ^0.1.1 (for `fitSync`).

### Removed

- `ml.mlp` (the hand-written network) and `ml.train` (its loop and
  callbacks). The callbacks live in `@tangent.to/nn` as `nn.callbacks`.

## [0.14.0] - 2026-09-02

### Added

- **Gaussian process: ARD by block.** `Matern` and `RBF` take `blocks`, mapping
  each input dimension to an entry of `lengthScale`, so a group of features
  shares one length scale. Three blocks cost three hyperparameters where
  per-feature ARD on 37 features cost 37. On a 340-row agronomic model that
  took the held-out RMSE from 9.9 to 9.1 and 95% interval coverage from 73%
  to 93%, with a learned noise that stayed put across folds where the
  per-feature version's collapsed to zero in four of seven. Forward, autodiff
  gradient, hyperparameter collection and `getParams` all follow the map.
- **Gaussian process: hyperparameter bounds.** `lengthScaleBounds`,
  `varianceBounds` (Matern, RBF) and `noiseLevelBounds` (WhiteKernel), as
  `[low, high]`, honoured by the optimizer in place of the hard-coded floors.
  A floor on the noise is the usual reason to set one.
- **`GaussianProcessRegressor.predictGradient(x)`.** The predictive mean and
  standard deviation at one input with their gradients in that input, from
  the autodiff path, compiled once per fit. What a gradient-based search over
  the input space needs. Stationary kernels only.

### Fixed

- A blocked `Matern` took the hand-derived likelihood gradient, which knows
  nothing of blocks: it indexed `lengthScale` by dimension, read `undefined`
  past the block count, and handed the optimizer NaN, which stopped at the
  initial values without a word. Blocked kernels go through the autodiff
  gradient.

## [0.13.1] - 2026-09-01

### Added

- `ordiplot` takes `minPredictorContribution`, a contribution threshold for an
  RDA's predictor arrows separate from `minLoadingContribution` (which now
  governs the response arrows). By default it inherits `minLoadingContribution`.

## [0.13.0] - 2026-09-01

### Added

- `ordiplot` accepts a fitted mva estimator (`ordiplot(pca, …)`) as well as
  its `.model`. An unfitted estimator says that `fit()` was never called.

### Fixed

- `PCA` warns when the `columns` it was given do not match the matrix width,
  instead of falling back to `var1..varN` in silence.

### Changed

- **BREAKING (visual) — `ordiplot`'s `loadingFactor` now defaults to `0` (auto)
  instead of `1`.** Arrows are fitted to 90% of the score cloud's radius, so
  the two halves of a biplot are readable against each other.

  A PCA hands back its halves on scales that differ by roughly √n: site scores
  are normalized to unit column norm, so each is of order 1/√n, while loadings
  stay of order 1. At the old default, and with `loadingScale`'s 3× on top, a
  333-row PCA drew arrows **13× longer** than the score cloud — the points
  rendered as a dot at the origin. A biplot whose two halves are not comparable
  is not really a biplot.

  Every existing biplot will change. Pass `loadingFactor: 1` to get the old
  lengths back. Note that `loadingScale` is inert under auto-scaling: the fit
  normalizes by the longest vector, which cancels any constant prefactor.


## [0.12.0] - 2026-09-01

### Added

- Autodiff gradients of the GP log marginal likelihood for every kernel (RBF,
  Matérn at any ν, RationalQuadratic, DotProduct, ConstantKernel, WhiteKernel,
  Periodic and their sums). Kernels other than Matérn previously fell back to a
  derivative-free search that stopped early.

### Fixed

- `new RBF([…], v)` and `new Matern([…], v)` discarded a positional ARD length
  scale and the variance, and fell back to the defaults.
- `RationalQuadratic`'s shape parameter is bounded during optimization, so it no
  longer runs off to ~1e13.

## [0.11.0] - 2026-09-01

### Added

- **`WhiteKernel`** — independent Gaussian noise on each observation,
  `k(x, x') = noiseLevel · δ(x, x')`. This is the *learnable* noise term: put it
  in the kernel and `optimize: true` estimates `noiseLevel` from the data, the
  way scikit-learn does. Exported from `@tangent.to/ds` alongside the other
  kernels.
- **Heteroscedastic observation noise for `GaussianProcessRegressor`** —
  `alpha` now accepts an array (or typed array) of per-observation noise
  variances, not just a scalar, matching sklearn's array-valued `alpha`. Use it
  when measurements differ in reliability: a poll weighted by sample size, a
  sensor with a per-reading error bar.

  ```js
  gp.fit(X, y, { alpha: polls.map((p) => p.samplingVariance) });
  ```

  It can also be passed to the constructor or in the declarative spec
  (`{ data, X, y, alpha }`). Length is validated against `y`, and the vector is
  used consistently by the fit factorization *and* by the hyperparameter
  optimizer's own likelihood — previously the optimizer would have tuned the
  kernel under a different noise model than the final fit used.

### Fixed

- **`GLM` ridge regularization no longer penalizes the intercept.** The penalty
  was applied to every column of the design matrix, intercept included. Since
  the intercept carries the scale of `y` rather than the influence of a
  predictor, shrinking it dragged the whole fit toward zero and the slopes
  distorted to compensate — on a target with mean ~100, `alpha: 1` already sent
  slopes of `3` and `-2` to `9.1` and `+5.0`, a sign flip. Coefficients now
  match `sklearn.linear_model.Ridge(fit_intercept=True)` to ~1e-14.

  Like sklearn, predictors are still not standardized, so the penalty remains
  scale-dependent — scale them yourself if you want uniform shrinkage.

  With `intercept: false` every column is penalized, as before.

### Changed

- **BREAKING — `GaussianProcessRegressor` no longer tunes `alpha`.** `alpha` is
  now strictly *known* noise, held fixed; a noise level to be *estimated*
  belongs in the kernel as a `WhiteKernel`. This is the split scikit-learn
  draws, and it removes the previous conflation where `alpha` silently played
  both roles.

  ```js
  // before — alpha was tuned by the optimizer
  new GaussianProcessRegressor({ kernel: new RBF(1, 1), alpha: 0.1, optimize: true })

  // after
  new GaussianProcessRegressor({
    kernel: new SumKernel({ kernels: [new RBF(1, 1), new WhiteKernel(0.1)] }),
    optimize: true,
  })
  ```

  The old form still runs — it simply leaves `alpha` at the value you gave it,
  with no error. Check any `optimize: true` call that relied on the noise being
  fitted.

  Note that the two are not interchangeable in one respect: a `WhiteKernel` is
  part of the kernel, so it enters the predictive variance and
  `predict(X, { returnStd: true })` returns the standard deviation of a new
  *observation*. Noise supplied through `alpha` gives the latent function's
  standard deviation instead. sklearn behaves the same way.

- `SumKernel.call()` now sums its children's covariance *matrices* rather than
  their pointwise `compute()` values. Identical numbers for kernels that are
  plain functions of the input values, but a `WhiteKernel` is not one: it has to
  know whether the matrix being built is `K(X, X)` or a cross-covariance
  `K(X₁, X₂)`. Summing per element would have dropped the noise term inside a
  sum, or leaked it into the train/test block.

- The analytic-gradient fast path (Matérn with ν ∈ {1.5, 2.5, ∞}, L-BFGS) now
  also covers a Matérn summed with a `WhiteKernel`, in either child order, so
  learning a noise level does not fall back to the derivative-free search.

[Unreleased]: https://github.com/tangent-to/ds/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/tangent-to/ds/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/tangent-to/ds/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/tangent-to/ds/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/tangent-to/ds/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/tangent-to/ds/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/tangent-to/ds/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/tangent-to/ds/releases/tag/v0.11.0
