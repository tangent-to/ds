# Changelog

Notable changes to `@tangent.to/ds`. This file starts at 0.11.0; for earlier
releases see the [git history](https://github.com/tangent-to/ds/commits/main)
and the [release tags](https://github.com/tangent-to/ds/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
