import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from loguru import logger

from titanicprediction.entities.core import Dataset, TrainedModel


class IDataRepository(Protocol):
    def load_data(self) -> Dataset: ...
    def save_data(self, dataset: Dataset) -> bool: ...
    def get_metadata(self) -> Dict[str, Any]: ...


class IModelRepository(Protocol):
    def save_model(self, model: TrainedModel, name: str) -> bool: ...
    def load_model(self, name: str) -> Optional[TrainedModel]: ...
    def list_models(self) -> List[str]: ...


class CSVDataRepository:
    def __init__(self, file_path: str, target_column: str = "Survived"):
        self.file_path = Path(file_path)
        self.target_column = target_column

    def load_data(self) -> Dataset:
        try:
            df = pd.read_csv(self.file_path)

            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")

            features = df.drop(columns=[self.target_column])
            target = df[self.target_column]

            return Dataset(
                features=features,
                target=target,
                feature_names=list(features.columns),
                target_name=self.target_column,
            )

        except Exception as e:
            logger.error(f"Error when loading data from {self.file_path}: {e}")
            raise

    def save_data(self, dataset: Dataset) -> bool:
        try:
            df = dataset.features.copy()
            if dataset.target is not None:
                df[dataset.target_name] = dataset.target

            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(self.file_path, index=False)
            logger.info(f"Data saved in {self.file_path}")
            return True

        except Exception as e:
            logger.error(f"Error when saving data to {self.file_path}: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        try:
            dataset = self.load_data()

            return {
                "file_path": str(self.file_path),
                "rows": len(dataset.features),
                "columns": len(dataset.features.columns),
                "feature_names": dataset.feature_names,
                "target_name": dataset.target_name,
                "missing_values": dataset.features.isnull().sum().to_dict(),
                "data_types": dataset.features.dtypes.astype(str).to_dict(),
            }
        except Exception as e:
            logger.error(f"Error when getting metadata: {e}")
            return {}


class FileModelRepository:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def save_model(self, model: TrainedModel, name: str) -> bool:
        try:
            model_file = self.models_dir / f"{name}.pkl"

            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model {name} saved in {model_file}")
            return True

        except Exception as e:
            logger.error(f"Error when saving model {name}: {e}")
            return False

    def load_model(self, name: str) -> Optional[TrainedModel]:
        try:
            model_file = self.models_dir / f"{name}.pkl"

            if not model_file.exists():
                logger.warning(f"File of model {model_file} not found")
                return None

            with open(model_file, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Model {name} loadded")
            return model

        except Exception as e:
            logger.error(f"Error when loading model {name}: {e}")
            return None

    def list_models(self) -> List[str]:
        try:
            model_files = list(self.models_dir.glob("*.pkl"))
            return [f.stem for f in model_files]
        except Exception as e:
            logger.error(f"Error when getting list of models: {e}")
            return []

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        model = self.load_model(name)
        if model is None:
            return None

        return {
            "name": name,
            "feature_names": model.feature_names,
            "metrics": model.metrics,
            "weights_shape": model.weights.shape if hasattr(model, "weights") else None,
        }

    def delete_model(self, name: str) -> bool:
        try:
            model_file = self.models_dir / f"{name}.pkl"

            if model_file.exists():
                model_file.unlink()
                logger.info(f"Model {name} deleted")
                return True
            else:
                logger.warning(f"Model {name} not found for deleting")
                return False

        except Exception as e:
            logger.error(f"Error when deleting model {name}: {e}")
            return False
