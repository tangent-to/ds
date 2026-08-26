"""
Compare ds GaussianProcessRegressor with scikit-learn's implementation.
Run this to get reference values for comparison.

IMPORTANT: Use optimizer=None to prevent sklearn from optimizing the kernel hyperparameters.
Otherwise sklearn changes length_scale and results won't match.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

np.random.seed(42)

# Test 1: RBF Kernel computation
print("=== Test 1: RBF Kernel Values ===")
kernel = RBF(length_scale=1.0)
kernel.length_scale_bounds = "fixed"  # Prevent optimization
X1 = np.array([[0], [1], [2]])
X2 = np.array([[0], [1], [2]])
K = kernel(X1, X2)
print("K(X, X) with length_scale=1.0:")
print(K)

# Test 2: Fit and predict
print("\n=== Test 2: Fit and Predict ===")
X_train = np.array([[0], [1], [2], [3], [4]]).reshape(-1, 1)
y_train = np.array([0, 1, 0, 1, 0])

# Use optimizer=None to disable hyperparameter optimization
gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1, optimizer=None, random_state=42)
gp.fit(X_train, y_train)

X_test = np.array([[0], [1], [2], [3], [4]])
y_pred, y_std = gp.predict(X_test, return_std=True)
print(f"Predictions: {y_pred}")
print(f"Std devs: {y_std}")

# Test 3: Predictions at training points should be close to training values
print("\n=== Test 3: Interpolation Accuracy ===")
print(f"Training y: {y_train}")
print(f"Predicted y: {y_pred}")
print(f"Max error: {np.max(np.abs(y_train - y_pred)):.6f}")

# Test 4: Kernel matrix properties
print("\n=== Test 4: Kernel Matrix Properties ===")
kernel2 = RBF(length_scale=2.0)
K2 = kernel2(X1, X2)
print("K(X, X) with length_scale=2.0:")
print(K2)

# Test 5: Variance at training points should be low
print("\n=== Test 5: Variance at Training Points ===")
gp_tight = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.01, optimizer=None, random_state=42)
gp_tight.fit(X_train, y_train)
_, y_std_tight = gp_tight.predict(X_test, return_std=True)
print(f"Std devs with alpha=0.01: {y_std_tight}")

# Test 6: Predict at new points
print("\n=== Test 6: Predict at New Points ===")
X_new = np.array([[0.5], [1.5], [2.5], [3.5]])
y_new, y_std_new = gp.predict(X_new, return_std=True)
print(f"Predictions at [0.5, 1.5, 2.5, 3.5]: {y_new}")
print(f"Std devs: {y_std_new}")

# Test 7: Sample from posterior
print("\n=== Test 7: Sample from Posterior ===")
# Note: sklearn uses random_state during fit, but sample_y uses its own random state
samples = gp.sample_y(X_test, n_samples=3, random_state=42)
print(f"Sample shape: {samples.shape}")
print(f"Sample 1: {samples[:, 0]}")
print(f"Sample 2: {samples[:, 1]}")
print(f"Sample 3: {samples[:, 2]}")

# Test 8: Heteroscedastic noise — alpha as an array of per-observation variances
# sklearn adds alpha[i] to K[i, i] instead of one scalar everywhere. This is the
# reference for ds's `fit(X, y, { alpha: [...] })`.
print("\n=== Test 8: Heteroscedastic alpha (array) ===")
X_het = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y_het = np.array([0.0, 0.8, 0.2, 1.4, 0.1, 1.1])
# Point 2 is nearly noise-free, point 4 is barely trusted at all.
alpha_het = np.array([0.01, 0.5, 0.001, 0.2, 2.0, 0.05])

gp_het = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.5), alpha=alpha_het, optimizer=None, random_state=42
)
gp_het.fit(X_het, y_het)

X_het_test = np.array([[0.0], [0.5], [1.0], [2.0], [2.5], [4.0], [5.0]])
mean_het, std_het = gp_het.predict(X_het_test, return_std=True)
print(f"X_test: {X_het_test.ravel().tolist()}")
print(f"mean:   {mean_het.tolist()}")
print(f"std:    {std_het.tolist()}")
print(f"log marginal likelihood: {gp_het.log_marginal_likelihood_value_!r}")

# A constant array must reproduce the scalar path exactly (backward compat).
gp_const_arr = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.5), alpha=np.full(6, 0.1), optimizer=None
).fit(X_het, y_het)
gp_scalar = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.5), alpha=0.1, optimizer=None
).fit(X_het, y_het)
print("constant-array == scalar:",
      np.allclose(gp_const_arr.predict(X_het_test), gp_scalar.predict(X_het_test)))

# Test 9: WhiteKernel — the LEARNABLE counterpart to alpha. sklearn keeps the
# two separate: alpha is fixed, WhiteKernel.noise_level is a hyperparameter, and
# the white term enters the predictive variance (alpha does not).
print("\n=== Test 9: WhiteKernel vs alpha ===")
from sklearn.gaussian_process.kernels import WhiteKernel

gp_white = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.5) + WhiteKernel(noise_level=0.1),
    alpha=0.0,
    optimizer=None,
)
gp_white.fit(X_het, y_het)
mean_w, std_w = gp_white.predict(X_het_test, return_std=True)
print(f"white mean: {mean_w.tolist()}")
print(f"white std:  {std_w.tolist()}")
print(f"white lml:  {gp_white.log_marginal_likelihood_value_!r}")

# Same posterior MEAN as alpha=0.1 (identical K + noise), but a larger std:
# WhiteKernel predicts a new observation, alpha the latent function.
gp_alpha = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.5), alpha=0.1, optimizer=None
).fit(X_het, y_het)
mean_a, std_a = gp_alpha.predict(X_het_test, return_std=True)
print("mean(white) == mean(alpha):", np.allclose(mean_w, mean_a))
print("std(white)^2 - std(alpha)^2:", (std_w**2 - std_a**2).tolist())
print(f"alpha std:  {std_a.tolist()}")
print(f"alpha lml:  {gp_alpha.log_marginal_likelihood_value_!r}")

print("\n=== Summary: Key Values for Comparison ===")
print("RBF(1.0) K[0,0]:", K[0, 0])
print("RBF(1.0) K[0,1]:", K[0, 1])
print("RBF(1.0) K[0,2]:", K[0, 2])
print("RBF(2.0) K[0,1]:", K2[0, 1])
print("Prediction at x=0:", y_pred[0])
print("Prediction at x=2:", y_pred[2])
print("Heteroscedastic mean:", repr(mean_het.tolist()))
print("Heteroscedastic std:", repr(std_het.tolist()))
