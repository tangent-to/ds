#!/usr/bin/env Rscript
#'
#' R Reference Implementations for @tangent.to/ds Testing
#'
#' Generates reference outputs using R's statistical packages:
#' - stats::glm() for GLM
#' - lme4::glmer()/lmer() for GLMM
#' - stats::prcomp() for PCA
#' - MASS::lda() for LDA
#'
#' Output: JSON file with reference results for comparison with JS implementations
#'

suppressPackageStartupMessages({
  library(jsonlite)
})

# Check for required packages
check_packages <- function() {
  required <- c("jsonlite", "MASS")
  optional <- c("lme4")

  missing_required <- required[!sapply(required, requireNamespace, quietly = TRUE)]
  if (length(missing_required) > 0) {
    stop(paste("Missing required packages:", paste(missing_required, collapse = ", ")))
  }

  has_lme4 <- requireNamespace("lme4", quietly = TRUE)
  return(list(has_lme4 = has_lme4))
}

packages <- check_packages()

# ==============================================================================
# GLM Tests
# ==============================================================================

test_gaussian_glm <- function() {
  # Simple linear regression via GLM
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(3, 5, 7, 6, 9, 11, 10, 13, 15, 17)

  df <- data.frame(y = y, X1 = X1, X2 = X2)
  model <- glm(y ~ X1 + X2, data = df, family = gaussian())

  list(
    test = "gaussian_glm",
    X = cbind(X1, X2),
    y = y,
    family = "gaussian",
    link = "identity",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    df_residual = df.residual(model)
  )
}

test_binomial_glm <- function() {
  # Logistic regression
  set.seed(42)
  n <- 50
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  # Create separable classes
  prob <- plogis(1 + 2*X1 - 1.5*X2)
  y <- rbinom(n, 1, prob)

  df <- data.frame(y = y, X1 = X1, X2 = X2)
  model <- glm(y ~ X1 + X2, data = df, family = binomial(link = "logit"))

  list(
    test = "binomial_glm",
    X = cbind(X1, X2),
    y = y,
    family = "binomial",
    link = "logit",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    accuracy = mean((fitted(model) > 0.5) == y)
  )
}

test_poisson_glm <- function() {
  # Poisson regression for count data
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(2, 3, 5, 4, 7, 9, 8, 11, 13, 15)

  df <- data.frame(y = y, X1 = X1, X2 = X2)
  model <- glm(y ~ X1 + X2, data = df, family = poisson(link = "log"))

  list(
    test = "poisson_glm",
    X = cbind(X1, X2),
    y = y,
    family = "poisson",
    link = "log",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model)
  )
}

# ==============================================================================
# GLMM Tests (if lme4 available)
# ==============================================================================

test_gaussian_glmm <- function() {
  if (!packages$has_lme4) return(NULL)
  library(lme4)

  # Random intercept model
  set.seed(123)
  n_groups <- 5
  n_per_group <- 10
  n <- n_groups * n_per_group

  group <- rep(letters[1:n_groups], each = n_per_group)
  X1 <- rnorm(n)
  group_effects <- rnorm(n_groups, sd = 2)
  y <- 2 + 1.5*X1 + group_effects[as.numeric(factor(group))] + rnorm(n, sd = 0.5)

  df <- data.frame(y = y, X1 = X1, group = group)
  model <- lmer(y ~ X1 + (1 | group), data = df, REML = FALSE)

  vc <- as.data.frame(VarCorr(model))

  list(
    test = "gaussian_glmm",
    X = as.numeric(X1),
    y = y,
    group = group,
    family = "gaussian",
    fixed_effects = as.numeric(fixef(model)),
    random_effects = as.numeric(ranef(model)$group[, 1]),
    variance_intercept = vc$vcov[1],
    variance_residual = vc$vcov[2],
    logLik = as.numeric(logLik(model)),
    aic = AIC(model)
  )
}

test_binomial_glmm <- function() {
  if (!packages$has_lme4) return(NULL)
  library(lme4)

  # Random intercept logistic regression
  set.seed(456)
  n_groups <- 4
  n_per_group <- 15
  n <- n_groups * n_per_group

  group <- rep(LETTERS[1:n_groups], each = n_per_group)
  X1 <- rnorm(n)
  group_effects <- c(-1, 0, 0.5, 1)
  prob <- plogis(0.5 + X1 + group_effects[as.numeric(factor(group))])
  y <- rbinom(n, 1, prob)

  df <- data.frame(y = y, X1 = X1, group = group)
  model <- glmer(y ~ X1 + (1 | group), data = df, family = binomial())

  vc <- as.data.frame(VarCorr(model))

  list(
    test = "binomial_glmm",
    X = as.numeric(X1),
    y = y,
    group = group,
    family = "binomial",
    fixed_effects = as.numeric(fixef(model)),
    random_effects = as.numeric(ranef(model)$group[, 1]),
    variance_intercept = vc$vcov[1],
    logLik = as.numeric(logLik(model)),
    aic = AIC(model),
    accuracy = mean((fitted(model) > 0.5) == y)
  )
}

# ==============================================================================
# PCA Test
# ==============================================================================

test_pca <- function() {
  # Simple PCA test
  set.seed(789)
  n <- 30

  # Create correlated data
  X1 <- rnorm(n)
  X2 <- X1 + rnorm(n, sd = 0.5)
  X3 <- rnorm(n)
  X4 <- X3 + rnorm(n, sd = 0.3)

  data_mat <- cbind(X1, X2, X3, X4)

  # PCA with centering only
  pca_fit <- prcomp(data_mat, center = TRUE, scale. = FALSE)

  # Eigenvalues from variance
  eigenvalues <- (pca_fit$sdev)^2
  variance_explained <- eigenvalues / sum(eigenvalues)

  list(
    test = "pca",
    data = data_mat,
    center = as.numeric(pca_fit$center),
    eigenvalues = eigenvalues,
    variance_explained = variance_explained,
    # Components are the rotation matrix (loadings)
    components = t(pca_fit$rotation),  # Transpose to match sklearn convention
    scores = pca_fit$x
  )
}

# ==============================================================================
# LDA Test
# ==============================================================================

test_lda <- function() {
  library(MASS)

  # Simple LDA test with 3 classes
  set.seed(101)
  n_per_class <- 20

  # Class 1: centered at (0, 0)
  X1_c1 <- rnorm(n_per_class, mean = 0, sd = 1)
  X2_c1 <- rnorm(n_per_class, mean = 0, sd = 1)

  # Class 2: centered at (3, 0)
  X1_c2 <- rnorm(n_per_class, mean = 3, sd = 1)
  X2_c2 <- rnorm(n_per_class, mean = 0, sd = 1)

  # Class 3: centered at (1.5, 3)
  X1_c3 <- rnorm(n_per_class, mean = 1.5, sd = 1)
  X2_c3 <- rnorm(n_per_class, mean = 3, sd = 1)

  X <- rbind(
    cbind(X1_c1, X2_c1),
    cbind(X1_c2, X2_c2),
    cbind(X1_c3, X2_c3)
  )
  colnames(X) <- c("X1", "X2")
  y <- factor(rep(c("A", "B", "C"), each = n_per_class))

  lda_fit <- lda(X, y)
  lda_pred <- predict(lda_fit, X)

  list(
    test = "lda",
    X = X,
    y = as.character(y),
    classes = levels(y),
    scaling = lda_fit$scaling,
    means = lda_fit$means,
    prior = as.numeric(lda_fit$prior),
    svd = lda_fit$svd,
    scores = lda_pred$x,
    predictions = as.character(lda_pred$class),
    accuracy = mean(lda_pred$class == y)
  )
}

# ==============================================================================
# Run All Tests
# ==============================================================================

run_all_tests <- function() {
  results <- list(
    gaussian_glm = test_gaussian_glm(),
    binomial_glm = test_binomial_glm(),
    poisson_glm = test_poisson_glm(),
    pca = test_pca(),
    lda = test_lda()
  )

  # Add GLMM tests if lme4 is available
  if (packages$has_lme4) {
    results$gaussian_glmm <- test_gaussian_glmm()
    results$binomial_glmm <- test_binomial_glmm()
    results$has_lme4 <- TRUE
  } else {
    results$has_lme4 <- FALSE
    message("Note: lme4 not available, GLMM tests skipped")
  }

  return(results)
}

# Main execution
args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) > 0) args[1] else "/tmp/r_comparison_results.json"

results <- run_all_tests()
json_output <- toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 15)
write(json_output, file = output_path)

cat(paste("R reference results written to", output_path, "\n"))
