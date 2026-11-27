"""Float64 NumPy reference for the shared tabular regression fixture."""
import json
import sys

import numpy as np


def regression(train, target, test, learning_rate, weights, test_target=None):
    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)
    scaled_train = (train - means) / scales
    scaled_test = (test - means) / scales
    target_mean = target.mean()
    target_scale = target.std(ddof=0)
    normalized_target = (target - target_mean) / target_scale

    design = np.column_stack((np.ones(len(train)), scaled_train))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    linear = np.column_stack((np.ones(len(test)), scaled_test)) @ coefficients

    w1, b1, w2, b2 = [np.array(w, dtype=np.float64) for w in weights]
    w1 = w1.reshape(2, 2)
    hidden = np.tanh(scaled_train @ w1 + b1)
    initial = hidden @ w2 + b2
    residual = initial - normalized_target
    d_output = 2 * residual / len(train)
    d_hidden = d_output[:, None] * w2 * (1 - hidden**2)
    w1 -= learning_rate * (scaled_train.T @ d_hidden)
    b1 -= learning_rate * d_hidden.sum(axis=0)
    w2 -= learning_rate * (hidden.T @ d_output)
    b2 -= learning_rate * d_output.sum()

    neural = (np.tanh(scaled_test @ w1 + b1) @ w2 + b2) * target_scale + target_mean
    result = {
        "means": means.tolist(),
        "scales": scales.tolist(),
        "scaledTrain": scaled_train.tolist(),
        "scaledTest": scaled_test.tolist(),
        "targetMean": float(target_mean),
        "targetScale": float(target_scale),
        "initialPrediction": initial.tolist(),
        "loss": float(np.mean(residual**2)),
        "updatedWeights": [w1.ravel().tolist(), b1.tolist(), w2.tolist(), b2.tolist()],
        "linearPrediction": linear.tolist(),
        "neuralPrediction": neural.tolist(),
    }
    if test_target is not None:
        result["rmse"] = [float(np.sqrt(np.mean((p - test_target)**2))) for p in (linear, neural)]
    return result


fixture = json.load(sys.stdin)
x = np.array(fixture["X"], dtype=np.float64)
y = np.array(fixture["y"], dtype=np.float64)
new_x = np.array(fixture["newX"], dtype=np.float64)
args = fixture["learningRate"], fixture["weights"]
results = [
    regression(x[train], y[train], x[test], *args, test_target=y[test])
    for train, test in ((f["trainIndices"], f["testIndices"]) for f in fixture["folds"])
]
results.append(regression(x, y, new_x, *args))
json.dump(results, sys.stdout, allow_nan=False)
