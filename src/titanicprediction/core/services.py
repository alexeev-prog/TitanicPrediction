import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from titanicprediction.core.algorithms import gradient_descent, predict, predict_proba
from titanicprediction.data.preprocessing import DataPreprocessor
from titanicprediction.entities.core import (
    Dataset,
    FeatureImpactAnalysis,
    Passenger,
    PredictionExplanation,
    TrainedModel,
)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    epochs: int = 1000
    test_size: float = 0.2
    random_state: int = 42
    convergence_tol: float = 1e-6
    lambda_reg: float = 0.01


@dataclass(frozen=True)
class TrainingResult:
    model: TrainedModel
    training_time: float
    final_loss: float
    metrics: dict[str, float]
    learning_curve: list[float]
    feature_importance: dict[str, float]
    config: TrainingConfig


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    passenger: Passenger
    probability: float
    prediction: bool
    confidence: float
    timestamp: datetime


@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: list[EvaluationResult]
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    std_accuracy: float
    std_precision: float
    std_recall: float
    std_f1: float


@dataclass(frozen=True)
class ConfidenceInterval:
    lower_bound: float
    upper_bound: float
    confidence_level: float


class IModelTrainingService(Protocol):
    def train_model(
        self, dataset: Dataset, config: TrainingConfig
    ) -> TrainingResult: ...

    def evaluate_model(
        self, model: TrainedModel, test_data: Dataset
    ) -> EvaluationResult: ...

    def cross_validate(
        self, dataset: Dataset, config: TrainingConfig, folds: int
    ) -> CrossValidationResult: ...


class IPredictionService(Protocol):
    def predict_survival(self, passenger: Passenger) -> PredictionResult: ...

    def batch_predict(self, passengers: list[Passenger]) -> list[PredictionResult]: ...

    def get_prediction_confidence(
        self, prediction: PredictionResult
    ) -> ConfidenceInterval: ...


class ModelTrainingService:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.poly_transformer = None
        self.X_mean = None
        self.X_std = None

    def train_model(self, dataset: Dataset, config: TrainingConfig) -> TrainingResult:
        start_time = time.time()

        processed_data = self.preprocessor.fit_transform(dataset)
        numeric_features = processed_data.features.select_dtypes(include=[np.number])
        X_train = numeric_features.values.astype(np.float64)

        original_numeric_feature_names = numeric_features.columns.tolist()

        self.poly_transformer = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=True
        )
        X_train_poly = self.poly_transformer.fit_transform(X_train)

        poly_feature_names = self.poly_transformer.get_feature_names_out(
            original_numeric_feature_names
        )

        self.X_mean = np.mean(X_train_poly, axis=0)
        self.X_std = np.std(X_train_poly, axis=0)
        self.X_std[self.X_std == 0] = 1
        X_train_normalized = (X_train_poly - self.X_mean) / self.X_std

        y_train = processed_data.target.values.astype(int)

        result = gradient_descent(
            X=X_train_normalized,
            y=y_train,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            convergence_tol=config.convergence_tol,
            lambda_reg=config.lambda_reg,
        )

        training_time = time.time() - start_time

        model = TrainedModel(
            weights=result.weights,
            bias=result.bias,
            feature_names=poly_feature_names.tolist(),
            training_metrics={},
            validation_metrics={},
            training_history=result.loss_history,
            model_config=config.__dict__,
            preprocessing_artifacts={
                "poly_transformer": self.poly_transformer,
                "X_mean": self.X_mean,
                "X_std": self.X_std,
                "original_feature_names": original_numeric_feature_names,
            },
        )

        return TrainingResult(
            model=model,
            training_time=training_time,
            final_loss=result.loss_history[-1],
            metrics={},
            learning_curve=result.loss_history,
            feature_importance=self._calculate_feature_importance(model),
            config=config,
        )

    def evaluate_model(
        self, model: TrainedModel, test_data: Dataset
    ) -> EvaluationResult:
        processed_test = self.preprocessor.transform(test_data)

        preprocessing_artifacts = model.preprocessing_artifacts
        self.poly_transformer = preprocessing_artifacts["poly_transformer"]
        self.X_mean = preprocessing_artifacts["X_mean"]
        self.X_std = preprocessing_artifacts["X_std"]

        numeric_test_features = processed_test.features.select_dtypes(
            include=[np.number]
        )
        X_test_original = numeric_test_features.values.astype(np.float64)

        if hasattr(self, "poly_transformer"):
            X_test_poly = self.poly_transformer.transform(X_test_original)
        else:
            X_test_poly = X_test_original

        if hasattr(self, "X_mean") and hasattr(self, "X_std"):
            X_test = (X_test_poly - self.X_mean) / self.X_std
        else:
            X_test = X_test_poly

        y_true = processed_test.target.values.astype(np.float64)

        if X_test.shape[1] != len(model.feature_names):
            aligned_X_test = np.zeros((X_test.shape[0], len(model.feature_names)))

            for i, _feature_name in enumerate(model.feature_names):
                if i < X_test.shape[1]:
                    aligned_X_test[:, i] = X_test[:, i]

            X_test = aligned_X_test

        predict_proba(X_test, model.weights, model.bias)
        y_pred = predict(X_test, model.weights, model.bias, threshold=0.5)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average="binary")
        recall = recall_score(y_true, y_pred, zero_division=0, average="binary")
        f1 = f1_score(y_true, y_pred, zero_division=0, average="binary")
        cm = confusion_matrix(y_true, y_pred)

        classification_report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": len(y_true),
        }

        return EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            classification_report=classification_report,
        )

    def _align_features_with_model(
        self, features: pd.DataFrame, model: TrainedModel
    ) -> pd.DataFrame:
        aligned_features = pd.DataFrame()

        for feature in model.feature_names:
            if feature in features.columns:
                aligned_features[feature] = features[feature]
            else:
                aligned_features[feature] = 0.0

        return aligned_features[model.feature_names]

    def cross_validate(
        self, dataset: Dataset, config: TrainingConfig, folds: int = 5
    ) -> CrossValidationResult:
        kf = KFold(n_splits=folds, shuffle=True, random_state=config.random_state)
        fold_results = []

        X = dataset.features.values
        y = dataset.target.values

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_dataset = Dataset(
                features=pd.DataFrame(X_train, columns=dataset.feature_names),
                target=pd.Series(y_train),
                feature_names=dataset.feature_names,
                target_name=dataset.target_name,
            )

            test_dataset = Dataset(
                features=pd.DataFrame(X_test, columns=dataset.feature_names),
                target=pd.Series(y_test),
                feature_names=dataset.feature_names,
                target_name=dataset.target_name,
            )

            training_result = self.train_model(train_dataset, config)
            eval_result = self.evaluate_model(training_result.model, test_dataset)
            fold_results.append(eval_result)

        accuracies = [r.accuracy for r in fold_results]
        precisions = [r.precision for r in fold_results]
        recalls = [r.recall for r in fold_results]
        f1_scores = [r.f1_score for r in fold_results]

        return CrossValidationResult(
            fold_results=fold_results,
            mean_accuracy=np.mean(accuracies),
            mean_precision=np.mean(precisions),
            mean_recall=np.mean(recalls),
            mean_f1=np.mean(f1_scores),
            std_accuracy=np.std(accuracies),
            std_precision=np.std(precisions),
            std_recall=np.std(recalls),
            std_f1=np.std(f1_scores),
        )

    def _calculate_feature_importance(self, model: TrainedModel) -> dict[str, float]:
        importance = {}
        total_importance = np.sum(np.abs(model.weights))

        for i, feature_name in enumerate(model.feature_names):
            importance[feature_name] = float(
                np.abs(model.weights[i]) / total_importance
            )

        return importance


class PredictionService:
    def __init__(self, model: TrainedModel, preprocessor: DataPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.preprocessing_artifacts = model.preprocessing_artifacts

    def predict_survival(self, passenger: Passenger) -> PredictionResult:
        try:
            passenger_df = self._passenger_to_dataframe(passenger)

            dummy_dataset = Dataset(
                features=passenger_df,
                target=None,
                feature_names=list(passenger_df.columns),
                target_name="dummy",
            )

            processed_data = self.preprocessor.transform(dummy_dataset)
            numeric_features = processed_data.features.select_dtypes(
                include=[np.number]
            )
            X_pred_original = numeric_features.values.astype(np.float64)

            if (
                self.preprocessing_artifacts
                and "poly_transformer" in self.preprocessing_artifacts
            ):
                poly_transformer = self.preprocessing_artifacts["poly_transformer"]
                X_pred_poly = poly_transformer.transform(X_pred_original)

                if (
                    "X_mean" in self.preprocessing_artifacts
                    and "X_std" in self.preprocessing_artifacts
                ):
                    X_mean = self.preprocessing_artifacts["X_mean"]
                    X_std = self.preprocessing_artifacts["X_std"]
                    X_pred = (X_pred_poly - X_mean) / X_std
                else:
                    X_pred = X_pred_poly
            else:
                X_pred = X_pred_original

            if X_pred.shape[1] != len(self.model.feature_names):
                X_pred = self._align_features(X_pred, self.model.feature_names)

            probability = predict_proba(X_pred, self.model.weights, self.model.bias)[0]

            if np.isnan(probability) or not np.isfinite(probability):
                probability = 0.5

            prediction = probability >= 0.5
            confidence = min(1.0, max(0.0, self._calculate_confidence(probability)))

            return PredictionResult(
                passenger=passenger,
                probability=float(probability),
                prediction=bool(prediction),
                confidence=confidence,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return PredictionResult(
                passenger=passenger,
                probability=0.5,
                prediction=False,
                confidence=0.5,
                timestamp=datetime.now(),
            )

    def _align_features(
        self, features: np.ndarray, expected_feature_names: list[str]
    ) -> np.ndarray:
        expected_count = len(expected_feature_names)
        current_count = features.shape[1]

        if current_count == expected_count:
            return features

        aligned_features = np.zeros((features.shape[0], expected_count))

        min_features = min(current_count, expected_count)
        aligned_features[:, :min_features] = features[:, :min_features]

        return aligned_features

    def batch_predict(self, passengers: list[Passenger]) -> list[PredictionResult]:
        return [self.predict_survival(passenger) for passenger in passengers]

    def get_prediction_confidence(
        self, prediction: PredictionResult
    ) -> ConfidenceInterval:
        probability = prediction.probability
        margin_of_error = 1.96 * np.sqrt((probability * (1 - probability)) / 100)

        return ConfidenceInterval(
            lower_bound=max(0.0, probability - margin_of_error),
            upper_bound=min(1.0, probability + margin_of_error),
            confidence_level=0.95,
        )

    def _passenger_to_dataframe(self, passenger: Passenger) -> pd.DataFrame:
        data = {
            "PassengerId": [passenger.passenger_id],
            "Pclass": [passenger.pclass],
            "Name": [passenger.name],
            "Sex": [passenger.sex],
            "Age": [passenger.age],
            "SibSp": [passenger.sibsp],
            "Parch": [passenger.parch],
            "Ticket": [passenger.ticket or "Unknown"],
            "Fare": [passenger.fare],
            "Cabin": [passenger.cabin or "Unknown"],
            "Embarked": [passenger.embarked],
        }
        return pd.DataFrame(data)

    def _calculate_confidence(self, probability: float) -> float:
        distance_from_decision = abs(probability - 0.5)
        confidence = 0.5 + distance_from_decision
        return float(confidence)


@dataclass
class ModelExplanationService:
    prediction_service: PredictionService

    def explain_prediction(self, passenger: Passenger) -> PredictionExplanation:
        prediction_result = self.prediction_service.predict_survival(passenger)
        feature_impacts = self._calculate_feature_impacts(passenger)
        decision_factors = self._extract_decision_factors(feature_impacts)
        confidence_level = self._determine_confidence_level(
            prediction_result.probability
        )

        return PredictionExplanation(
            prediction=prediction_result.prediction,
            probability=prediction_result.probability,
            feature_impacts=feature_impacts,
            decision_factors=decision_factors,
            confidence_level=confidence_level,
        )

    def get_model_statistics(self, model: TrainedModel) -> dict[str, Any]:
        weights = model.weights
        return {
            "total_features": len(weights),
            "weight_magnitude": float(np.linalg.norm(weights)),
            "positive_weights": int(np.sum(weights > 0)),
            "negative_weights": int(np.sum(weights < 0)),
            "weight_range": {"min": float(weights.min()), "max": float(weights.max())},
            "bias": float(model.bias),
            "weight_mean": float(weights.mean()),
            "weight_std": float(weights.std()),
        }

    def _calculate_feature_impacts(
        self, passenger: Passenger
    ) -> list[FeatureImpactAnalysis]:
        model = self.prediction_service.model
        preprocessor = self.prediction_service.preprocessor

        passenger_df = self.prediction_service._passenger_to_dataframe(passenger)
        dummy_dataset = Dataset(
            features=passenger_df,
            target=None,
            feature_names=list(passenger_df.columns),
            target_name="dummy",
        )

        processed_data = preprocessor.transform(dummy_dataset)
        numeric_features = processed_data.features.select_dtypes(include=[np.number])
        X_original = numeric_features.values.astype(np.float64)

        preprocessing_artifacts = model.preprocessing_artifacts
        if preprocessing_artifacts and "poly_transformer" in preprocessing_artifacts:
            poly_transformer = preprocessing_artifacts["poly_transformer"]
            X_poly = poly_transformer.transform(X_original)

            if (
                "X_mean" in preprocessing_artifacts
                and "X_std" in preprocessing_artifacts
            ):
                X_mean = preprocessing_artifacts["X_mean"]
                X_std = preprocessing_artifacts["X_std"]
                X = (X_poly - X_mean) / X_std
            else:
                X = X_poly
        else:
            X = X_original

        if X.shape[1] != len(model.feature_names):
            aligned_X = np.zeros((X.shape[0], len(model.feature_names)))
            min_features = min(X.shape[1], len(model.feature_names))
            aligned_X[:, :min_features] = X[:, :min_features]
            X = aligned_X

        feature_impacts = []
        for i, feature_name in enumerate(model.feature_names):
            if i < len(model.weights):
                feature_value = X[0][i]
                weight = model.weights[i]
                impact = weight * feature_value
                contribution = abs(impact) / (abs(np.dot(X[0], model.weights)) + 1e-10)

                feature_impacts.append(
                    FeatureImpactAnalysis(
                        feature_name=feature_name,
                        impact_score=float(impact),
                        weight=float(weight),
                        feature_value=float(feature_value),
                        contribution=float(contribution),
                    )
                )

        return sorted(feature_impacts, key=lambda x: abs(x.impact_score), reverse=True)

    def _extract_decision_factors(
        self, feature_impacts: list[FeatureImpactAnalysis]
    ) -> list[str]:
        factors = []
        top_impacts = feature_impacts[:5]

        for impact in top_impacts:
            if impact.impact_score > 0:
                factors.append(f"{impact.feature_name} increased survival chance")
            else:
                factors.append(f"{impact.feature_name} decreased survival chance")

        return factors

    def _determine_confidence_level(self, probability: float) -> str:
        if probability > 0.8 or probability < 0.2:
            return "High"
        if probability > 0.7 or probability < 0.3:
            return "Medium"
        return "Low"


class ServiceFactory:
    @staticmethod
    def create_training_service(preprocessor: DataPreprocessor) -> ModelTrainingService:
        return ModelTrainingService(preprocessor)

    @staticmethod
    def create_prediction_service(
        model: TrainedModel, preprocessor: DataPreprocessor
    ) -> PredictionService:
        return PredictionService(model, preprocessor)

    @staticmethod
    def create_explanation_service(
        prediction_service: PredictionService,
    ) -> ModelExplanationService:
        return ModelExplanationService(prediction_service)
