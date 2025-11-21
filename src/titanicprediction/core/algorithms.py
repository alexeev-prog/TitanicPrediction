from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

FloatArray = np.ndarray


@dataclass
class GradientDescentResult:
    weights: FloatArray
    bias: float
    loss_history: List[float]
    convergence_info: Dict[str, Any]


def sigmoid(z: FloatArray) -> FloatArray:
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def add_polynomial_features(X: FloatArray, degree: int = 2) -> FloatArray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)


def binary_cross_entropy(
    y_true: FloatArray,
    y_pred: FloatArray,
    weights: FloatArray,
    lambda_reg: float = 0.01,
    epsilon: float = 1e-12,
) -> float:
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos > 0 and n_neg > 0:
        weight_pos = n_neg / len(y_true)
        weight_neg = n_pos / len(y_true)
        weights_class = np.where(y_true == 1, weight_pos, weight_neg)

        loss = -np.mean(
            weights_class
            * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )
        reg_loss = (lambda_reg / (2 * len(y_true))) * np.sum(weights**2)
        return float(loss + reg_loss)
    else:
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        reg_loss = (lambda_reg / (2 * len(y_true))) * np.sum(weights**2)
        return float(loss + reg_loss)


def compute_gradients(
    X: FloatArray, y_true: FloatArray, y_pred: FloatArray, class_weight: dict = None
) -> Tuple[FloatArray, float]:
    m = len(y_true)
    error = y_pred - y_true

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos > 0 and n_neg > 0:
        weight_neg = n_pos / len(y_true)
        weight_pos = n_neg / len(y_true)

        weights = np.where(y_true == 1, weight_pos, weight_neg)
        error = error * weights

        dw = (1 / np.sum(weights)) * np.dot(X.T, error)
        db = (1 / np.sum(weights)) * np.sum(error)
    else:
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)

    return dw, float(db)


def gradient_descent(
    X: FloatArray,
    y: FloatArray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    convergence_tol: float = 1e-6,
    beta: float = 0.9,
    lambda_reg: float = 0.01,
) -> GradientDescentResult:
    n_features = X.shape[1]

    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    loss_history = []

    v_dw = np.zeros_like(weights)
    v_db = 0.0

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)

        loss = binary_cross_entropy(y, predictions, weights, lambda_reg)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, predictions)

        dw_reg = dw + (lambda_reg / len(y)) * weights

        v_dw = beta * v_dw + (1 - beta) * dw_reg
        v_db = beta * v_db + (1 - beta) * db

        weights -= learning_rate * v_dw
        bias -= learning_rate * v_db

        if epoch > 100 and len(loss_history) > 1:
            loss_change = abs(loss_history[-2] - loss_history[-1])
            if loss_change < convergence_tol:
                break

    convergence_info = {
        "final_loss": loss_history[-1] if loss_history else float("inf"),
        "epochs_completed": len(loss_history),
        "learning_rate": learning_rate,
        "with_momentum": True,
    }

    return GradientDescentResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        convergence_info=convergence_info,
    )


def predict_proba(X: FloatArray, weights: FloatArray, bias: float) -> FloatArray:
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def predict(
    X: FloatArray, weights: FloatArray, bias: float, threshold: float = None
) -> FloatArray:
    probabilities = predict_proba(X, weights, bias)

    if threshold is None:
        threshold = 0.35

    return (probabilities >= threshold).astype(int)
