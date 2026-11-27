# Base R double-precision reference for the shared tabular regression fixture.
fixture <- jsonlite::fromJSON(file("stdin"), simplifyVector = FALSE)
row_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
x <- row_matrix(fixture$X)
y <- unlist(fixture$y)
new_x <- row_matrix(fixture$newX)
lr <- fixture$learningRate
weights <- lapply(fixture$weights, unlist)

regression <- function(train, target, test, test_target = NULL) {
  means <- colMeans(train)
  scales <- sqrt(colMeans(sweep(train, 2, means)^2))
  scaled_train <- scale(train, center = means, scale = scales)
  scaled_test <- scale(test, center = means, scale = scales)
  target_mean <- mean(target)
  target_scale <- sqrt(mean((target - target_mean)^2))
  normalized_target <- (target - target_mean) / target_scale

  coefficients <- lm.fit(cbind(1, scaled_train), target)$coefficients
  linear <- drop(cbind(1, scaled_test) %*% coefficients)

  w1 <- matrix(weights[[1]], nrow = 2, byrow = TRUE)
  b1 <- weights[[2]]
  w2 <- weights[[3]]
  b2 <- weights[[4]]
  hidden <- tanh(sweep(scaled_train %*% w1, 2, b1, "+"))
  initial <- drop(hidden %*% w2) + b2
  residual <- initial - normalized_target
  d_output <- 2 * residual / nrow(train)
  d_hidden <- outer(d_output, w2) * (1 - hidden^2)
  w1 <- w1 - lr * crossprod(scaled_train, d_hidden)
  b1 <- b1 - lr * colSums(d_hidden)
  w2 <- w2 - lr * drop(crossprod(hidden, d_output))
  b2 <- b2 - lr * sum(d_output)

  neural <- (drop(tanh(sweep(scaled_test %*% w1, 2, b1, "+")) %*% w2) + b2) *
    target_scale + target_mean
  result <- list(
    means = means,
    scales = scales,
    scaledTrain = scaled_train,
    scaledTest = scaled_test,
    targetMean = jsonlite::unbox(target_mean),
    targetScale = jsonlite::unbox(target_scale),
    initialPrediction = initial,
    loss = jsonlite::unbox(mean(residual^2)),
    updatedWeights = list(as.vector(t(w1)), b1, w2, b2),
    linearPrediction = linear,
    neuralPrediction = neural
  )
  if (!is.null(test_target)) {
    result$rmse <- c(sqrt(mean((linear - test_target)^2)), sqrt(mean((neural - test_target)^2)))
  }
  result
}

results <- lapply(fixture$folds, function(fold) {
  train <- unlist(fold$trainIndices) + 1L
  test <- unlist(fold$testIndices) + 1L
  regression(x[train, , drop = FALSE], y[train], x[test, , drop = FALSE], y[test])
})
results[[length(results) + 1L]] <- regression(x, y, new_x)
cat(jsonlite::toJSON(results, digits = NA))
