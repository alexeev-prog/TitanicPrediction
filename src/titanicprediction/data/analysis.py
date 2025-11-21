from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from titanicprediction.entities.core import Dataset, TrainedModel


class PlotType(Enum):
    HISTOGRAM = "histogram"
    BAR = "bar"
    SCATTER = "scatter"
    BOX = "box"
    HEATMAP = "heatmap"
    LINE = "line"
    PIE = "pie"
    KDE = "kde"
    VIOLIN = "violin"
    COUNT = "count"


class BackendType(Enum):
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"


@dataclass
class PlotConfig:
    plot_type: PlotType
    title: str
    x_label: str
    y_label: str
    figsize: Tuple[int, int] = (10, 6)
    style: str = "seaborn-v0_8"
    colors: List[str] = None
    save_format: str = "png"
    dpi: int = 300
    font_size: int = 12
    legend: bool = True
    grid: bool = True


@dataclass
class PlotResult:
    figure: plt.Figure
    axes: plt.Axes
    config: PlotConfig
    metadata: Dict[str, Any]


class IVisualizer(Protocol):
    def create_plot(self, data: Any, plot_config: PlotConfig) -> PlotResult: ...

    def save_plot(self, plot: PlotResult, filename: str) -> Path: ...
    def get_supported_formats(self) -> List[str]: ...


class DistributionVisualizer:
    def create_plot(self, data: Any, plot_config: PlotConfig) -> PlotResult:
        plt.style.use(plot_config.style)
        fig, ax = plt.subplots(figsize=plot_config.figsize)

        if plot_config.plot_type == PlotType.HISTOGRAM:
            self._create_histogram(ax, data, plot_config)
        elif plot_config.plot_type == PlotType.KDE:
            self._create_kde(ax, data, plot_config)
        elif plot_config.plot_type == PlotType.VIOLIN:
            self._create_violin(ax, data, plot_config)

        self._apply_common_styling(ax, plot_config)

        metadata = {
            "data_points": len(data) if hasattr(data, "__len__") else None,
            "data_type": type(data).__name__,
            "created_at": pd.Timestamp.now().isoformat(),
        }

        return PlotResult(figure=fig, axes=ax, config=plot_config, metadata=metadata)

    def _create_histogram(
        self, ax: plt.Axes, data: pd.Series, config: PlotConfig
    ) -> None:
        color = config.colors[0] if config.colors else "skyblue"
        ax.hist(data.dropna(), bins=30, alpha=0.7, color=color, edgecolor="black")
        ax.set_xlabel(config.x_label, fontsize=config.font_size)
        ax.set_ylabel("Frequency", fontsize=config.font_size)

    def _create_kde(self, ax: plt.Axes, data: pd.Series, config: PlotConfig) -> None:
        color = config.colors[0] if config.colors else "red"
        data_clean = data.dropna()
        data_clean.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_xlabel(config.x_label, fontsize=config.font_size)
        ax.set_ylabel("Density", fontsize=config.font_size)

    def _create_violin(self, ax: plt.Axes, data: pd.Series, config: PlotConfig) -> None:
        config.colors[0] if config.colors else "lightgreen"
        ax.violinplot(data.dropna(), showmeans=True, showmedians=True)
        ax.set_ylabel(config.y_label, fontsize=config.font_size)

    def _apply_common_styling(self, ax: plt.Axes, config: PlotConfig) -> None:
        ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")
        if config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.font_size - 2)


class CorrelationVisualizer:
    def _apply_common_styling(self, ax: plt.Axes, config: PlotConfig) -> None:
        ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")
        if config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.font_size - 2)

    def create_plot(self, data: pd.DataFrame, plot_config: PlotConfig) -> PlotResult:
        plt.style.use(plot_config.style)
        fig, ax = plt.subplots(figsize=plot_config.figsize)

        if plot_config.plot_type == PlotType.HEATMAP:
            self._create_correlation_heatmap(ax, data, plot_config)

        self._apply_common_styling(ax, plot_config)

        metadata = {
            "correlation_matrix_shape": data.shape,
            "features": list(data.columns),
            "created_at": pd.Timestamp.now().isoformat(),
        }

        return PlotResult(figure=fig, axes=ax, config=plot_config, metadata=metadata)

    def _create_correlation_heatmap(
        self, ax: plt.Axes, corr_matrix: pd.DataFrame, config: PlotConfig
    ) -> None:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")


class TrainingVisualizer:
    def create_plot(
        self, loss_history: List[float], plot_config: PlotConfig
    ) -> PlotResult:
        plt.style.use(plot_config.style)
        fig, ax = plt.subplots(figsize=plot_config.figsize)

        if plot_config.plot_type == PlotType.LINE:
            self._create_training_curve(ax, loss_history, plot_config)

        self._apply_common_styling(ax, plot_config)

        metadata = {
            "training_epochs": len(loss_history),
            "final_loss": loss_history[-1] if loss_history else None,
            "min_loss": min(loss_history) if loss_history else None,
            "created_at": pd.Timestamp.now().isoformat(),
        }

        return PlotResult(figure=fig, axes=ax, config=plot_config, metadata=metadata)

    def _create_training_curve(
        self, ax: plt.Axes, loss_history: List[float], config: PlotConfig
    ) -> None:
        color = config.colors[0] if config.colors else "blue"
        ax.plot(range(len(loss_history)), loss_history, color=color, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=config.font_size)
        ax.set_ylabel("Loss", fontsize=config.font_size)
        ax.set_yscale("log")


class FeatureAnalysisVisualizer:
    def create_plot(self, dataset: Dataset, plot_config: PlotConfig) -> PlotResult:
        plt.style.use(plot_config.style)
        fig, ax = plt.subplots(figsize=plot_config.figsize)

        if plot_config.plot_type == PlotType.BAR:
            if "target" in str(type(dataset)).lower():
                self._create_survival_by_feature(ax, dataset, plot_config)
            else:
                self._create_feature_distribution(ax, dataset, plot_config)
        elif plot_config.plot_type == PlotType.BOX:
            self._create_box_plot(ax, dataset, plot_config)
        elif plot_config.plot_type == PlotType.SCATTER:
            self._create_scatter_plot(ax, dataset, plot_config)
        elif plot_config.plot_type == PlotType.PIE:
            self._create_pie_chart(ax, dataset, plot_config)
        elif plot_config.plot_type == PlotType.COUNT:
            self._create_count_plot(ax, dataset, plot_config)

        self._apply_common_styling(ax, plot_config)

        metadata = {
            "dataset_shape": dataset.features.shape,
            "feature_names": dataset.feature_names[:5],
            "created_at": pd.Timestamp.now().isoformat(),
        }

        return PlotResult(figure=fig, axes=ax, config=plot_config, metadata=metadata)

    def _create_survival_by_feature(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        feature_name = config.x_label.lower()
        if feature_name in dataset.features.columns:
            survival_by_feature = dataset.features.groupby(feature_name)[
                dataset.target_name
            ].agg(["mean", "count"])
            colors = config.colors if config.colors else ["#ff9999", "#66b3ff"]

            x_pos = np.arange(len(survival_by_feature))
            ax.bar(
                x_pos,
                survival_by_feature["mean"] * 100,
                color=colors[0],
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_xlabel(config.x_label, fontsize=config.font_size)
            ax.set_ylabel("Survival Rate (%)", fontsize=config.font_size)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(x) for x in survival_by_feature.index])

            for i, (idx, row) in enumerate(survival_by_feature.iterrows()):
                ax.text(
                    i,
                    row["mean"] * 100 + 1,
                    f"{row['mean'] * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=config.font_size - 2,
                )

    def _create_feature_distribution(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        feature_name = config.x_label.lower()
        if feature_name in dataset.features.columns:
            feature_data = dataset.features[feature_name].value_counts().sort_index()
            colors = (
                config.colors
                if config.colors
                else plt.cm.Set3(np.linspace(0, 1, len(feature_data)))
            )

            ax.bar(
                range(len(feature_data)),
                feature_data.values,
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_xlabel(config.x_label, fontsize=config.font_size)
            ax.set_ylabel("Count", fontsize=config.font_size)
            ax.set_xticks(range(len(feature_data)))
            ax.set_xticklabels([str(x) for x in feature_data.index], rotation=45)

    def _create_box_plot(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        feature_name = config.x_label.lower()
        target_name = config.y_label.lower()

        if (
            feature_name in dataset.features.columns
            and target_name in dataset.features.columns
        ):
            data_to_plot = []
            groups = dataset.features[target_name].unique()

            for group in sorted(groups):
                group_data = dataset.features[dataset.features[target_name] == group][
                    feature_name
                ].dropna()
                data_to_plot.append(group_data)

            ax.boxplot(data_to_plot, labels=[str(g) for g in groups])
            ax.set_xlabel(target_name, fontsize=config.font_size)
            ax.set_ylabel(feature_name, fontsize=config.font_size)

    def _create_scatter_plot(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        x_feature = config.x_label.lower()
        y_feature = config.y_label.lower()

        if (
            x_feature in dataset.features.columns
            and y_feature in dataset.features.columns
        ):
            colors = config.colors if config.colors else ["red", "blue"]

            if dataset.target is not None:
                for target_val in [0, 1]:
                    mask = dataset.target == target_val
                    ax.scatter(
                        dataset.features[mask][x_feature],
                        dataset.features[mask][y_feature],
                        c=colors[target_val],
                        alpha=0.6,
                        label=f"Survived={target_val}",
                    )
                if config.legend:
                    ax.legend()
            else:
                ax.scatter(
                    dataset.features[x_feature], dataset.features[y_feature], alpha=0.6
                )

            ax.set_xlabel(x_feature, fontsize=config.font_size)
            ax.set_ylabel(y_feature, fontsize=config.font_size)

    def _create_pie_chart(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        feature_name = config.x_label.lower()
        if feature_name in dataset.features.columns:
            value_counts = dataset.features[feature_name].value_counts()
            colors = (
                config.colors
                if config.colors
                else plt.cm.Pastel1(range(len(value_counts)))
            )

            wedges, texts, autotexts = ax.pie(
                value_counts.values,
                labels=value_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontsize(config.font_size - 2)

    def _create_count_plot(
        self, ax: plt.Axes, dataset: Dataset, config: PlotConfig
    ) -> None:
        feature_name = config.x_label.lower()
        if feature_name in dataset.features.columns and dataset.target is not None:
            cross_tab = pd.crosstab(dataset.features[feature_name], dataset.target)
            colors = config.colors if config.colors else ["#ff6b6b", "#4ecdc4"]

            bar_width = 0.35
            x_pos = np.arange(len(cross_tab))

            for i, target_val in enumerate([0, 1]):
                ax.bar(
                    x_pos + i * bar_width,
                    cross_tab[target_val],
                    bar_width,
                    label=f"Survived={target_val}",
                    color=colors[i],
                    alpha=0.7,
                )

            ax.set_xlabel(feature_name, fontsize=config.font_size)
            ax.set_ylabel("Count", fontsize=config.font_size)
            ax.set_xticks(x_pos + bar_width / 2)
            ax.set_xticklabels([str(x) for x in cross_tab.index])

            if config.legend:
                ax.legend()

    def _apply_common_styling(self, ax: plt.Axes, config: PlotConfig) -> None:
        ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")
        if config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.font_size - 2)


class EDAVisualizer:
    def __init__(self, output_dir: str = "data/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.distribution_visualizer = DistributionVisualizer()
        self.correlation_visualizer = CorrelationVisualizer()
        self.training_visualizer = TrainingVisualizer()
        self.feature_visualizer = FeatureAnalysisVisualizer()

    def create_survival_analysis_plots(self, dataset: Dataset) -> Dict[str, Path]:
        plots = {}

        plot_df = dataset.features.copy()
        if dataset.target is not None:
            plot_df["Survived"] = dataset.target

        if dataset.target is not None:
            config = PlotConfig(
                plot_type=PlotType.PIE,
                title="Survival Distribution on Titanic",
                x_label="Survived",
                y_label="Percentage",
                figsize=(8, 8),
                colors=["#ff6b6b", "#4ecdc4"],
            )

            fig, ax = plt.subplots(figsize=config.figsize)
            survival_counts = dataset.target.value_counts()
            labels = ["Did Not Survive", "Survived"]
            colors = config.colors if config.colors else ["#ff6b6b", "#4ecdc4"]

            wedges, texts, autotexts = ax.pie(
                survival_counts.values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")

            plot_result = PlotResult(figure=fig, axes=ax, config=config, metadata={})
            plots["survival_distribution"] = self._save_plot(
                plot_result, "survival_distribution"
            )

        if "Pclass" in plot_df.columns and "Survived" in plot_df.columns:
            config = PlotConfig(
                plot_type=PlotType.BAR,
                title="Survival Rate by Passenger Class",
                x_label="Pclass",
                y_label="Survival Rate (%)",
                colors=["#ff9999", "#66b3ff", "#99ff99"],
            )

            fig, ax = plt.subplots(figsize=config.figsize)

            survival_by_class = plot_df.groupby("Pclass")["Survived"].mean() * 100

            ax.bar(
                range(len(survival_by_class)),
                survival_by_class.values,
                color=config.colors[: len(survival_by_class)],
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_xlabel("Passenger Class", fontsize=config.font_size)
            ax.set_ylabel("Survival Rate (%)", fontsize=config.font_size)
            ax.set_xticks(range(len(survival_by_class)))
            ax.set_xticklabels([f"Class {cls}" for cls in survival_by_class.index])
            ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")

            for i, value in enumerate(survival_by_class.values):
                ax.text(
                    i,
                    value + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=config.font_size - 2,
                )

            plot_result = PlotResult(figure=fig, axes=ax, config=config, metadata={})
            plots["survival_by_class"] = self._save_plot(
                plot_result, "survival_by_class"
            )

        return plots

    def create_demographic_plots(self, dataset: Dataset) -> Dict[str, Path]:
        plots = {}

        plot_df = dataset.features.copy()
        if dataset.target is not None:
            plot_df["Survived"] = dataset.target

        if "Sex" in plot_df.columns and "Survived" in plot_df.columns:
            config = PlotConfig(
                plot_type=PlotType.BAR,
                title="Survival Rate by Gender",
                x_label="Sex",
                y_label="Survival Rate (%)",
                colors=["#ff6b6b", "#4ecdc4"],
            )

            fig, ax = plt.subplots(figsize=config.figsize)

            survival_by_sex = plot_df.groupby("Sex")["Survived"].mean() * 100

            ax.bar(
                range(len(survival_by_sex)),
                survival_by_sex.values,
                color=config.colors[: len(survival_by_sex)],
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_xlabel("Gender", fontsize=config.font_size)
            ax.set_ylabel("Survival Rate (%)", fontsize=config.font_size)
            ax.set_xticks(range(len(survival_by_sex)))
            ax.set_xticklabels([sex.capitalize() for sex in survival_by_sex.index])
            ax.set_title(config.title, fontsize=config.font_size + 2, fontweight="bold")

            for i, value in enumerate(survival_by_sex.values):
                ax.text(
                    i,
                    value + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=config.font_size - 2,
                )

            plot_result = PlotResult(figure=fig, axes=ax, config=config, metadata={})
            plots["survival_by_gender"] = self._save_plot(
                plot_result, "survival_by_gender"
            )

        if "Age" in plot_df.columns:
            config = PlotConfig(
                plot_type=PlotType.HISTOGRAM,
                title="Age Distribution of Passengers",
                x_label="Age",
                y_label="Frequency",
                colors=["skyblue"],
            )
            plot_result = self.distribution_visualizer.create_plot(
                plot_df["Age"].dropna(), config
            )
            plots["age_distribution"] = self._save_plot(plot_result, "age_distribution")

        return plots

    def create_correlation_analysis(self, dataset: Dataset) -> Dict[str, Path]:
        plots = {}

        numeric_features = dataset.features.select_dtypes(include=[np.number])
        correlation_matrix = numeric_features.corr()

        config = PlotConfig(
            plot_type=PlotType.HEATMAP,
            title="Correlation Matrix of Numerical Features",
            x_label="Features",
            y_label="Features",
            figsize=(10, 8),
        )

        plot_result = self.correlation_visualizer.create_plot(
            correlation_matrix, config
        )
        plots["correlation_matrix"] = self._save_plot(plot_result, "correlation_matrix")

        return plots

    def create_training_plots(self, loss_history: List[float]) -> Dict[str, Path]:
        plots = {}

        config = PlotConfig(
            plot_type=PlotType.LINE,
            title="Model Training Loss Over Epochs",
            x_label="Epoch",
            y_label="Loss",
            colors=["blue"],
        )

        plot_result = self.training_visualizer.create_plot(loss_history, config)
        plots["training_curve"] = self._save_plot(plot_result, "training_curve")

        return plots

    def create_feature_importance_plot(self, model: TrainedModel) -> Dict[str, Path]:
        plots = {}

        importance_df = pd.DataFrame(
            {"feature": model.feature_names, "importance": np.abs(model.weights)}
        ).sort_values("importance", ascending=True)

        config = PlotConfig(
            plot_type=PlotType.BAR,
            title="Feature Importance",
            x_label="Importance",
            y_label="Features",
            figsize=(10, 8),
            colors=["lightsteelblue"],
        )

        temp_dataset = Dataset(
            features=importance_df,
            target=None,
            feature_names=["feature", "importance"],
            target_name="importance",
        )

        plot_result = self.feature_visualizer.create_plot(temp_dataset, config)
        plots["feature_importance"] = self._save_plot(plot_result, "feature_importance")

        return plots

    def _save_plot(self, plot_result: PlotResult, filename: str) -> Path:
        filepath = self.output_dir / f"{filename}.{plot_result.config.save_format}"
        plot_result.figure.savefig(
            filepath, dpi=plot_result.config.dpi, bbox_inches="tight", facecolor="white"
        )
        plt.close(plot_result.figure)
        logger.info(f"Plot saved: {filepath}")
        return filepath

    def get_supported_formats(self) -> List[str]:
        return ["png", "jpg", "jpeg", "svg", "pdf"]
