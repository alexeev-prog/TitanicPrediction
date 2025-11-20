from enum import Enum
from dataclasses import dataclass
from typing import Union, Literal, Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np


AgeType = Union[float, None]
SurvivalType = Union[bool, None]
ProbabilityType = float
PercentageType = float

SexType = Literal["male", "female"]
PclassType = Literal[1, 2, 3]
EmbarkedType = Literal["C", "Q", "S"]
TitleType = Literal["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Other"]


class Gender(Enum):
    male = 0
    female = 1


@dataclass
class Passenger:
    passenger_id: int
    survived: SurvivalType
    pclass: PclassType
    name: str
    sex: SexType
    age: AgeType
    sibsp: int
    parch: int
    fare: float
    embarked: Optional[EmbarkedType]
    cabin: Optional[str]
    title: Optional[TitleType]
    ticket: Optional[str]

    def validate(self) -> bool:
        if not self.name or not self.sex:
            return False

        if self.age is not None and (self.age < 0 or self.age > 120):
            return False

        if self.fare < 0:
            return False

        if self.sibsp < 0 or self.parch < 0:
            return False

        if self.pclass not in [1, 2, 3]:
            return False

        return True

    def is_valid(self) -> bool:
        required_fields = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

        for field in required_fields:
            value = getattr(self, field)
            if value is None:
                return False

        return self.age is not None and self.fare is not None

    def get_missing_fields(self) -> List[str]:
        missing = []

        if not self.name:
            missing.append("name")
        if not self.sex:
            missing.append("sex")
        if self.age is None:
            missing.append("age")
        if self.fare is None:
            missing.append("fare")
        if self.embarked is None:
            missing.append("embarked")

        if self.age is not None and (self.age < 0 or self.age > 120):
            missing.append("age (invalid range)")
        if self.fare < 0:
            missing.append("fare (negative)")
        if self.sibsp < 0:
            missing.append("sibsp (negative)")
        if self.parch < 0:
            missing.append("parch (negative)")

        return missing


@dataclass
class Dataset:
    features: pd.DataFrame
    target: Optional[pd.Series]
    feature_names: List[str]
    target_name: Optional[str]
    metadata: Dict[str, Any] = None

    def get_shape(self) -> Tuple[int, int]:
        return self.features.shape

    def get_feature_types(self) -> Dict[str, str]:
        return self.features.dtypes.astype(str).to_dict()

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=1 - ratio, random_state=42
        )

        train_dataset = Dataset(
            features=X_train,
            target=y_train,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

        test_dataset = Dataset(
            features=X_test,
            target=y_test,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

        return train_dataset, test_dataset

    def describe(self) -> Dict[str, Any]:
        return {
            "shape": self.get_shape(),
            "feature_types": self.get_feature_types(),
            "target_distribution": self.target.value_counts().to_dict()
            if self.target is not None
            else {},
            "missing_values": self.features.isnull().sum().to_dict(),
        }


@dataclass
class TrainedModel:
    weights: np.ndarray
    bias: float
    feature_names: List[str]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    training_history: List[float]
    model_config: Dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """С УЛУЧШЕННОЙ ВАЛИДАЦИЕЙ"""
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: model expects {len(self.feature_names)} features, "
                f"but got {features.shape[1]}. Feature names: {self.feature_names}"
            )

        linear_output = np.dot(features, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-np.clip(linear_output, -500, 500)))
        predictions = (probabilities >= 0.5).astype(int)

        return predictions

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of positive class for given features"""
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )

        linear_output = np.dot(features, self.weights) + self.bias

        probabilities = 1 / (1 + np.exp(-np.clip(linear_output, -500, 500)))

        return np.column_stack([1 - probabilities, probabilities])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on absolute weights"""
        if len(self.weights) != len(self.feature_names):
            raise ValueError("Weights and feature_names length mismatch")

        absolute_weights = np.abs(self.weights)
        total_importance = np.sum(absolute_weights)

        if total_importance > 0:
            importance_dict = {
                feature: float(weight / total_importance * 100)
                for feature, weight in zip(self.feature_names, absolute_weights)
            }
        else:
            importance_dict = {feature: 0.0 for feature in self.feature_names}

        return importance_dict
