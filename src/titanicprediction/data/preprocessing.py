import pandas as pd
from typing import Protocol, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import re
from loguru import logger
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer

from titanicprediction.entities.core import Dataset


class IDataTransformer(Protocol):
    def fit(self, dataset: Dataset) -> None: ...
    def transform(self, dataset: Dataset) -> Dataset: ...
    def fit_transform(self, dataset: Dataset) -> Dataset: ...
    def get_params(self) -> Dict[str, Any]: ...


@dataclass
class AgeImputer:
    strategy: Literal["mean", "median", "mode", "constant"] = "median"
    fill_value: Optional[float] = None
    _imputer: Any = None

    def fit(self, dataset: Dataset) -> None:
        if "Age" not in dataset.features.columns:
            return

        age_data = dataset.features["Age"].values.reshape(-1, 1)

        if self.strategy == "constant" and self.fill_value is not None:
            self._imputer = SimpleImputer(
                strategy="constant", fill_value=self.fill_value
            )
        else:
            self._imputer = SimpleImputer(strategy=self.strategy)

        self._imputer.fit(age_data)

    def transform(self, dataset: Dataset) -> Dataset:
        if self._imputer is None or "Age" not in dataset.features.columns:
            return dataset

        features = dataset.features.copy()
        age_data = features["Age"].values.reshape(-1, 1)
        imputed_age = self._imputer.transform(age_data)
        features["Age"] = imputed_age.flatten()

        return Dataset(
            features=features,
            target=dataset.target,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "fill_value": self.fill_value}


@dataclass
class CategoricalEncoder:
    encoding_type: Literal["onehot", "label"] = "onehot"
    columns: List[str] = None
    handle_unknown: Literal["error", "ignore", "use_encoded_value"] = "error"
    _encoders: Dict[str, Any] = None
    _feature_names: List[str] = None

    def __post_init__(self):
        self._encoders = {}
        self._feature_names = []
        if self.columns is None:
            self.columns = []

    def fit(self, dataset: Dataset) -> None:
        self._encoders = {}
        self._feature_names = dataset.feature_names.copy()

        for column in self.columns:
            if column not in dataset.features.columns:
                continue

            if self.encoding_type == "onehot":
                unique_values = dataset.features[column].dropna().unique()
                self._encoders[column] = list(unique_values)
            else:
                encoder = LabelEncoder()
                encoder.fit(dataset.features[column].dropna())
                self._encoders[column] = encoder

    def transform(self, dataset: Dataset) -> Dataset:
        features = dataset.features.copy()

        for column in self.columns:
            if column not in features.columns or column not in self._encoders:
                continue

            if self.encoding_type == "onehot":
                unique_values = self._encoders[column]
                for value in unique_values:
                    new_col_name = f"{column}_{value}"
                    features[new_col_name] = (features[column] == value).astype(int)

                features = features.drop(columns=[column])
                self._update_feature_names(column, unique_values)
            else:
                encoder = self._encoders[column]
                try:
                    encoded = encoder.transform(features[column])
                    features[column] = encoded
                except ValueError as e:
                    if self.handle_unknown == "error":
                        raise e
                    else:
                        features[column] = -1

        return Dataset(
            features=features,
            target=dataset.target,
            feature_names=self._feature_names,
            target_name=dataset.target_name,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

    def get_params(self) -> Dict[str, Any]:
        return {
            "encoding_type": self.encoding_type,
            "columns": self.columns,
            "handle_unknown": self.handle_unknown,
        }

    def _update_feature_names(self, original_col: str, new_cols: List[str]) -> None:
        if original_col in self._feature_names:
            index = self._feature_names.index(original_col)
            self._feature_names.remove(original_col)

            new_names = [f"{original_col}_{col}" for col in new_cols]
            for i, new_name in enumerate(new_names):
                self._feature_names.insert(index + i, new_name)


@dataclass
class FeatureScaler:
    method: Literal["standard", "minmax", "robust"] = "standard"
    columns: List[str] = None
    with_mean: bool = True
    with_std: bool = True
    _scalers: Dict[str, Any] = None

    def __post_init__(self):
        self._scalers = {}
        if self.columns is None:
            self.columns = []

    def fit(self, dataset: Dataset) -> None:
        self._scalers = {}

        for column in self.columns:
            if column not in dataset.features.columns:
                continue

            data = dataset.features[column].values.reshape(-1, 1)

            if self.method == "standard":
                scaler = StandardScaler(
                    with_mean=self.with_mean, with_std=self.with_std
                )
            elif self.method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            scaler.fit(data)
            self._scalers[column] = scaler

    def transform(self, dataset: Dataset) -> Dataset:
        features = dataset.features.copy()

        for column in self.columns:
            if column not in features.columns or column not in self._scalers:
                continue

            scaler = self._scalers[column]
            data = features[column].values.reshape(-1, 1)
            scaled_data = scaler.transform(data)
            features[column] = scaled_data.flatten()

        return Dataset(
            features=features,
            target=dataset.target,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "columns": self.columns,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
        }


@dataclass
class TitleExtractor:
    name_column: str = "Name"
    title_column: str = "Title"
    custom_mappings: Dict[str, str] = None
    _title_patterns: Dict[str, str] = None

    def __post_init__(self):
        if self.custom_mappings is None:
            self.custom_mappings = {}

        self._title_patterns = {
            "Mr": r"\bMr\.",
            "Mrs": r"\bMrs\.",
            "Miss": r"\bMiss\.",
            "Master": r"\bMaster\.",
            "Dr": r"\bDr\.",
            "Rev": r"\bRev\.",
        }

    def fit(self, dataset: Dataset) -> None:
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        features = dataset.features.copy()

        if self.name_column not in features.columns:
            return dataset

        def extract_title(name):
            if pd.isna(name):
                return "Unknown"

            for title, pattern in self._title_patterns.items():
                if re.search(pattern, name, re.IGNORECASE):
                    return title

            return "Other"

        features[self.title_column] = features[self.name_column].apply(extract_title)

        for original, mapped in self.custom_mappings.items():
            features[self.title_column] = features[self.title_column].replace(
                original, mapped
            )

        new_feature_names = dataset.feature_names + [self.title_column]

        return Dataset(
            features=features,
            target=dataset.target,
            feature_names=new_feature_names,
            target_name=dataset.target_name,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        return self.transform(dataset)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name_column": self.name_column,
            "title_column": self.title_column,
            "custom_mappings": self.custom_mappings,
        }


class DataPreprocessor:
    def __init__(self):
        self.preprocessing_steps: List[Tuple[str, IDataTransformer]] = []
        self.fitted: bool = False

    def add_step(self, name: str, transformer: IDataTransformer) -> None:
        self.preprocessing_steps.append((name, transformer))

    def fit(self, dataset: Dataset) -> None:
        current_dataset = dataset

        for name, transformer in self.preprocessing_steps:
            logger.info(f"Fitting transformer: {name}")
            transformer.fit(current_dataset)

        self.fitted = True

    def transform(self, dataset: Dataset) -> Dataset:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transformation")

        current_dataset = dataset

        for name, transformer in self.preprocessing_steps:
            logger.info(f"Applying transformer: {name}")
            current_dataset = transformer.transform(current_dataset)

        return current_dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        dummy_dataset = Dataset(
            features=features,
            target=None,
            feature_names=list(features.columns),
            target_name="dummy",
        )

        transformed = self.transform(dummy_dataset)
        return transformed.features

    def get_params(self) -> Dict[str, Any]:
        params = {}
        for name, transformer in self.preprocessing_steps:
            params[name] = transformer.get_params()
        return params


@dataclass
class ColumnDropper:
    columns: List[str] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []

    def fit(self, dataset: Dataset) -> None:
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        features = dataset.features.copy()

        columns_to_drop = [col for col in self.columns if col in features.columns]
        if columns_to_drop:
            features = features.drop(columns=columns_to_drop)
            new_feature_names = [
                name for name in dataset.feature_names if name not in columns_to_drop
            ]
        else:
            new_feature_names = dataset.feature_names

        return Dataset(
            features=features,
            target=dataset.target,
            feature_names=new_feature_names,
            target_name=dataset.target_name,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        return self.transform(dataset)

    def get_params(self) -> Dict[str, Any]:
        return {"columns": self.columns}


class PreprocessorFactory:
    @staticmethod
    def create_titanic_preprocessor() -> DataPreprocessor:
        preprocessor = DataPreprocessor()

        preprocessor.add_step(
            "title_extractor",
            TitleExtractor(
                name_column="Name",
                title_column="Title",
                custom_mappings={"Ms": "Mrs", "Mlle": "Miss", "Mme": "Mrs"},
            ),
        )

        preprocessor.add_step("age_imputer", AgeImputer(strategy="median"))

        preprocessor.add_step(
            "drop_passenger_id",
            ColumnDropper(columns=["PassengerId", "Name", "Ticket", "Cabin"]),
        )

        preprocessor.add_step(
            "categorical_encoder",
            CategoricalEncoder(
                encoding_type="onehot",
                columns=["Sex", "Embarked", "Title"],
                handle_unknown="ignore",
            ),
        )

        preprocessor.add_step(
            "feature_scaler",
            FeatureScaler(
                method="standard",
                columns=["Age", "Fare", "SibSp", "Parch", "Pclass"],
                with_mean=True,
                with_std=True,
            ),
        )

        return preprocessor
