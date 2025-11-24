from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

FloatArray = np.ndarray


@dataclass
class GradientDescentResult:
    """
    Gradient descent result dataclass state.

    Attributes:
        weights: Final weights after gradient descent optimization.
        bias: Final bias term after optimization.
        loss_history: List of loss values during training.
        convergence_info: Dictionary containing convergence information.

    """

    weights: FloatArray
    bias: float
    loss_history: list[float]
    convergence_info: dict[str, Any]


def sigmoid(z: FloatArray) -> FloatArray:
    """
    Sigmoid function.

    Create the sigmoid function from float array.

    Args:
        z: Input array.

    Returns:
        Sigmoid function result.

    """
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def add_polynomial_features(x: FloatArray, degree: int = 2) -> FloatArray:
    """
    Add polynomial features to the input data.

    Args:
        x: Input feature matrix.
        degree: Degree of polynomial features. Defaults to 2.

    Returns:
        Transformed feature matrix with polynomial features.

    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(x)


def binary_cross_entropy(
    y_true: FloatArray,
    y_pred: FloatArray,
    weights: FloatArray,
    lambda_reg: float = 0.01,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate binary cross-entropy loss with regularization.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        weights: Model weights for regularization.
        lambda_reg: Regularization parameter. Defaults to 0.01.
        epsilon: Small value to avoid numerical issues. Defaults to 1e-12.

    Returns:
        Binary cross-entropy loss value with regularization.

    """
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
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg_loss = (lambda_reg / (2 * len(y_true))) * np.sum(weights**2)
    return float(loss + reg_loss)


def compute_gradients(
    x: FloatArray,
    y_true: FloatArray,
    y_pred: FloatArray,
    class_weight: dict | None = None,
) -> tuple[FloatArray, float]:
    """
    Compute gradients for logistic regression.

    Args:
        x: Input feature matrix.
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        class_weight: Optional class weights dictionary.

    Returns:
        Tuple containing:
            - Weight gradients
            - Bias gradient

    """
    m = len(y_true)
    error = y_pred - y_true

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos > 0 and n_neg > 0:
        weight_neg = n_pos / len(y_true)
        weight_pos = n_neg / len(y_true)

        weights = np.where(y_true == 1, weight_pos, weight_neg)
        error = error * weights

        dw = (1 / np.sum(weights)) * np.dot(x.T, error)
        db = (1 / np.sum(weights)) * np.sum(error)
    else:
        dw = (1 / m) * np.dot(x.T, error)
        db = (1 / m) * np.sum(error)

    return dw, float(db)


def gradient_descent(
    x: FloatArray,
    y: FloatArray,
    learning_rate: float = 0.001,
    epochs: int = 1000,
    convergence_tol: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    lambda_reg: float = 0.01,
) -> GradientDescentResult:
    """
    Perform gradient descent optimization with Adam optimizer.

    Args:
        x: Input feature matrix.
        y: Target labels.
        learning_rate: Learning rate for optimization. Defaults to 0.001.
        epochs: Maximum number of epochs. Defaults to 1000.
        convergence_tol: Convergence tolerance. Defaults to 1e-6.
        beta1: Adam beta1 parameter. Defaults to 0.9.
        beta2: Adam beta2 parameter. Defaults to 0.999.
        epsilon: Adam epsilon parameter. Defaults to 1e-8.
        lambda_reg: Regularization parameter. Defaults to 0.01.

    Returns:
        GradientDescentResult containing optimization results.

    """
    n_features = x.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    loss_history = []

    m_dw = np.zeros_like(weights)
    m_db = 0.0
    v_dw = np.zeros_like(weights)
    v_db = 0.0

    for epoch in range(1, epochs + 1):
        z = np.dot(x, weights) + bias
        predictions = sigmoid(z)

        loss = binary_cross_entropy(y, predictions, weights, lambda_reg)
        loss_history.append(loss)

        dw, db = compute_gradients(x, y, predictions)
        dw_reg = dw + (lambda_reg / len(y)) * weights

        m_dw = beta1 * m_dw + (1 - beta1) * dw_reg
        m_db = beta1 * m_db + (1 - beta1) * db
        v_dw = beta2 * v_dw + (1 - beta2) * (dw_reg**2)
        v_db = beta2 * v_db + (1 - beta2) * (db**2)

        m_dw_corrected = m_dw / (1 - beta1**epoch)
        m_db_corrected = m_db / (1 - beta1**epoch)
        v_dw_corrected = v_dw / (1 - beta2**epoch)
        v_db_corrected = v_db / (1 - beta2**epoch)

        weights -= learning_rate * (
            m_dw_corrected / (np.sqrt(v_dw_corrected) + epsilon)
        )
        bias -= learning_rate * (m_db_corrected / (np.sqrt(v_db_corrected) + epsilon))

        if epoch > 100 and abs(loss_history[-2] - loss_history[-1]) < convergence_tol:
            break

    convergence_info = {
        "final_loss": loss_history[-1],
        "epochs_completed": len(loss_history),
        "learning_rate": learning_rate,
        "optimizer": "adam",
    }

    return GradientDescentResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        convergence_info=convergence_info,
    )


def standard_gradient_descent(
    x: FloatArray,
    y: FloatArray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    convergence_tol: float = 1e-6,
    beta: float = 0.9,
    lambda_reg: float = 0.01,
) -> GradientDescentResult:
    """
    Perform standard gradient descent with momentum.

    Args:
        x: Input feature matrix.
        y: Target labels.
        learning_rate: Learning rate for optimization. Defaults to 0.01.
        epochs: Maximum number of epochs. Defaults to 1000.
        convergence_tol: Convergence tolerance. Defaults to 1e-6.
        beta: Momentum parameter. Defaults to 0.9.
        lambda_reg: Regularization parameter. Defaults to 0.01.

    Returns:
        GradientDescentResult containing optimization results.

    """
    n_features = x.shape[1]

    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    loss_history = []

    v_dw = np.zeros_like(weights)
    v_db = 0.0

    for epoch in range(epochs):
        z = np.dot(x, weights) + bias
        predictions = sigmoid(z)

        loss = binary_cross_entropy(y, predictions, weights, lambda_reg)
        loss_history.append(loss)

        dw, db = compute_gradients(x, y, predictions)

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


def predict_proba(x: FloatArray, weights: FloatArray, bias: float) -> FloatArray:
    """
    Predict probabilities for binary classification.

    Args:
        x: Input feature matrix.
        weights: Model weights.
        bias: Model bias.

    Returns:
        Predicted probabilities for positive class.

    """
    z = np.dot(x, weights) + bias
    return sigmoid(z)


def predict(
    x: FloatArray, weights: FloatArray, bias: float, threshold: float | None = None
) -> FloatArray:
    """
    Make binary predictions.

    Args:
        x: Input feature matrix.
        weights: Model weights.
        bias: Model bias.
        threshold: Classification threshold. Defaults to 0.35.

    Returns:
        Binary predictions (0 or 1).

    """
    probabilities = predict_proba(x, weights, bias)

    if threshold is None:
        threshold = 0.35

    return (probabilities >= threshold).astype(int)
