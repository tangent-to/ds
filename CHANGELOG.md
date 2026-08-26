# Changelog

Notable changes to `@tangent.to/ds`. This file starts at 0.11.0; for earlier
releases see the [git history](https://github.com/tangent-to/ds/commits/main)
and the [release tags](https://github.com/tangent-to/ds/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
