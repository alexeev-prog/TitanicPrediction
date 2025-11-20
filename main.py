import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

from titanicprediction.data.repositories import CSVDataRepository, FileModelRepository
from titanicprediction.data.preprocessing import PreprocessorFactory, DataPreprocessor
from titanicprediction.core.services import (
    ServiceFactory,
    ModelTrainingService,
    PredictionService,
)
from titanicprediction.data.analysis import EDAVisualizer
from titanicprediction.interface.web import TitanicApp


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class StreamlitConfig:
    port: int = 8501
    host: str = "localhost"
    theme: str = "light"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    page_title: str = "Titanic Survival Predictor"
    page_icon: str = "🚢"


@dataclass
class MLPipelineConfig:
    data_path: str = "datasets/TitanicDataset.csv"
    models_dir: str = "models"
    plots_dir: str = "plots"
    default_learning_rate: float = 0.01
    default_epochs: int = 1000
    default_test_size: float = 0.2
    random_state: int = 42


@dataclass
class UIConfig:
    show_technical_details: bool = True
    enable_data_export: bool = True
    max_prediction_history: int = 10
    auto_generate_plots: bool = True
    cache_timeout: int = 3600


@dataclass
class AppConfig:
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = "logs/titanic_ml.log"
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    ml_pipeline: MLPipelineConfig = field(default_factory=MLPipelineConfig)
    ui: UIConfig = field(default_factory=UIConfig)


@dataclass
class DependencyContainer:
    data_repository: CSVDataRepository
    model_repository: FileModelRepository
    training_service: ModelTrainingService
    visualizer: EDAVisualizer
    preprocessor: DataPreprocessor
    prediction_service: Optional[PredictionService] = None


class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized: bool = False

    def register_service(self, name: str, service: Any) -> None:
        self._services[name] = service

    def get_service(self, name: str) -> Any:
        if name not in self._services:
            raise KeyError(f"Service '{name}' not found in registry")
        return self._services[name]

    def get_all_services(self) -> Dict[str, Any]:
        return self._services.copy()

    def is_initialized(self) -> bool:
        return self._initialized

    def mark_initialized(self) -> None:
        self._initialized = True


class ConfigurationManager:
    def __init__(self):
        self._config: Optional[AppConfig] = None
        self._config_paths: List[Path] = []

    def load_configuration(self, config_path: Optional[str] = None) -> AppConfig:
        default_config = self._get_default_config()

        if config_path and Path(config_path).exists():
            file_config = self._load_config_from_file(Path(config_path))
            default_config = self._merge_configs(default_config, file_config)

        env_config = self._load_config_from_env()
        default_config = self._merge_configs(default_config, env_config)

        self._config = default_config
        self._validate_configuration()

        return self._config

    def get_config(self) -> AppConfig:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def _get_default_config(self) -> AppConfig:
        return AppConfig()

    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                    return yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    return json.load(f)
                else:
                    logging.warning(
                        f"Unsupported config file format: {config_path.suffix}"
                    )
                    return {}
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}")
            return {}

    def _load_config_from_env(self) -> Dict[str, Any]:
        env_config = {}

        if os.getenv("TITANIC_ML_ENV"):
            env_config["environment"] = Environment(os.getenv("TITANIC_ML_ENV"))

        if os.getenv("TITANIC_ML_LOG_LEVEL"):
            env_config["log_level"] = LogLevel(os.getenv("TITANIC_ML_LOG_LEVEL"))

        if os.getenv("TITANIC_ML_DATA_PATH"):
            env_config["ml_pipeline"] = {"data_path": os.getenv("TITANIC_ML_DATA_PATH")}

        return env_config

    def _merge_configs(self, base: AppConfig, overlay: Dict[str, Any]) -> AppConfig:
        base_dict = self._config_to_dict(base)
        merged_dict = self._deep_merge(base_dict, overlay)
        return self._dict_to_config(merged_dict)

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(config)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        streamlit_config = StreamlitConfig(**config_dict.get("streamlit", {}))
        ml_config = MLPipelineConfig(**config_dict.get("ml_pipeline", {}))
        ui_config = UIConfig(**config_dict.get("ui", {}))

        return AppConfig(
            environment=Environment(config_dict.get("environment", "development")),
            log_level=LogLevel(config_dict.get("log_level", "INFO")),
            log_file=config_dict.get("log_file"),
            streamlit=streamlit_config,
            ml_pipeline=ml_config,
            ui=ui_config,
        )

    def _deep_merge(
        self, base: Dict[str, Any], overlay: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_configuration(self) -> None:
        config = self._config

        if not Path(config.ml_pipeline.data_path).exists():
            logging.warning(f"Data path does not exist: {config.ml_pipeline.data_path}")

        if config.streamlit.port < 1024 or config.streamlit.port > 65535:
            raise ValueError(f"Invalid port number: {config.streamlit.port}")

        if config.ml_pipeline.default_learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if config.ml_pipeline.default_epochs <= 0:
            raise ValueError("Number of epochs must be positive")


class ApplicationInitializer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.service_registry = ServiceRegistry()
        self.dependency_container: Optional[DependencyContainer] = None

    def setup_dependencies(self) -> DependencyContainer:
        logging.info("Setting up application dependencies...")

        data_repository = CSVDataRepository(
            file_path=self.config.ml_pipeline.data_path, target_column="Survived"
        )

        model_repository = FileModelRepository(
            models_dir=self.config.ml_pipeline.models_dir
        )

        preprocessor = PreprocessorFactory.create_titanic_preprocessor()

        training_service = ServiceFactory.create_training_service(preprocessor)

        visualizer = EDAVisualizer(output_dir=self.config.ml_pipeline.plots_dir)

        self.dependency_container = DependencyContainer(
            data_repository=data_repository,
            model_repository=model_repository,
            training_service=training_service,
            visualizer=visualizer,
            preprocessor=preprocessor,
        )

        self._register_services()

        logging.info("Dependencies setup completed successfully")
        return self.dependency_container

    def _register_services(self) -> None:
        if self.dependency_container is None:
            return

        self.service_registry.register_service(
            "data_repository", self.dependency_container.data_repository
        )
        self.service_registry.register_service(
            "model_repository", self.dependency_container.model_repository
        )
        self.service_registry.register_service(
            "training_service", self.dependency_container.training_service
        )
        self.service_registry.register_service(
            "visualizer", self.dependency_container.visualizer
        )
        self.service_registry.register_service(
            "preprocessor", self.dependency_container.preprocessor
        )

        self.service_registry.mark_initialized()

    def initialize_services(self) -> None:
        logging.info("Initializing services...")

        self._create_directories()
        self._warmup_services()

        logging.info("Services initialization completed")

    def _create_directories(self) -> None:
        directories = [
            Path(self.config.ml_pipeline.models_dir),
            Path(self.config.ml_pipeline.plots_dir),
            Path("logs"),
            Path("data/raw"),
            Path("data/processed"),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

    def _warmup_services(self) -> None:
        try:
            data_repo = self.service_registry.get_service("data_repository")
            dataset = data_repo.load_data()
            logging.info(f"Data warmup: loaded {len(dataset.features)} rows")

            visualizer = self.service_registry.get_service("visualizer")
            visualizer.create_survival_analysis_plots(dataset)
            logging.info("Visualization warmup: created survival analysis plots")

        except Exception as e:
            logging.warning(f"Service warmup failed: {e}")

    def get_service_registry(self) -> ServiceRegistry:
        return self.service_registry

    def get_dependency_container(self) -> DependencyContainer:
        if self.dependency_container is None:
            raise RuntimeError("Dependencies not setup")
        return self.dependency_container


class ApplicationRunner:
    def __init__(self, config: AppConfig, service_registry: ServiceRegistry):
        self.config = config
        self.service_registry = service_registry

    def run(self) -> None:
        logging.info("Starting Titanic ML Application...")

        self._setup_streamlit_config()

        app = TitanicApp()

        logging.info(
            f"Application started on http://{self.config.streamlit.host}:{self.config.streamlit.port}"
        )

        try:
            app.run()
        except KeyboardInterrupt:
            logging.info("Application stopped by user")
        except Exception as e:
            logging.error(f"Application error: {e}")
            raise
        finally:
            self._cleanup()

    def _setup_streamlit_config(self) -> None:
        import streamlit as st

        st.set_page_config(
            page_title=self.config.streamlit.page_title,
            page_icon=self.config.streamlit.page_icon,
            layout=self.config.streamlit.layout,
            initial_sidebar_state=self.config.streamlit.initial_sidebar_state,
        )

    def _cleanup(self) -> None:
        logging.info("Cleaning up application resources...")


def setup_logging(config: AppConfig) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=config.log_level.value, format=log_format, handlers=handlers
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Titanic Survival Predictor ML Application"
    )

    parser.add_argument(
        "--config", type=str, help="Path to configuration file", default=None
    )

    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing"],
        help="Application environment",
        default=None,
    )

    parser.add_argument(
        "--data-path", type=str, help="Path to Titanic dataset", default=None
    )

    parser.add_argument("--port", type=int, help="Streamlit port", default=None)

    parser.add_argument("--host", type=str, help="Streamlit host", default=None)

    return parser.parse_args()


def apply_cli_args_to_config(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    if args.environment:
        config.environment = Environment(args.environment)

    if args.data_path:
        config.ml_pipeline.data_path = args.data_path

    if args.port:
        config.streamlit.port = args.port

    if args.host:
        config.streamlit.host = args.host

    return config


def main() -> None:
    args = parse_arguments()

    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(args.config)

    config = apply_cli_args_to_config(args, config)

    setup_logging(config)

    logging.info(f"Starting Titanic ML Application in {config.environment.value} mode")

    try:
        initializer = ApplicationInitializer(config)
        initializer.setup_dependencies()
        initializer.initialize_services()

        service_registry = initializer.get_service_registry()

        runner = ApplicationRunner(config, service_registry)
        runner.run()

    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
