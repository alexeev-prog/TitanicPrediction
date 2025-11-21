import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from titanicprediction.core.services import ServiceFactory, TrainingConfig
from titanicprediction.data.analysis import EDAVisualizer
from titanicprediction.data.preprocessing import PreprocessorFactory
from titanicprediction.data.repositories import CSVDataRepository
from titanicprediction.entities.core import Dataset, Passenger, TrainedModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AppState(TypedDict):
    current_page: str
    dataset: Optional[Dataset]
    trained_model: Optional[TrainedModel]
    preprocessing_pipeline: Optional[Any]
    current_predictions: List[Any]
    training_history: List[float]
    training_result: Optional[Any]


class DataTableComponent:
    def render(
        self, data: pd.DataFrame, title: str = "Data Table", page_size: int = 10
    ) -> None:
        st.subheader(title)

        col1, col2 = st.columns([2, 1])

        with col1:
            search_term = st.text_input("Search data...", key=f"search_{title}")

        with col2:
            sort_column = st.selectbox("Sort by", data.columns, key=f"sort_{title}")

        if search_term:
            filtered_data = data[
                data.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False).any(), axis=1
                )
            ]
        else:
            filtered_data = data

        if sort_column in filtered_data.columns:
            filtered_data = filtered_data.sort_values(by=sort_column)

        st.dataframe(filtered_data, width="stretch")

        total_pages = max(1, len(filtered_data) // page_size)
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1, key=f"page_{title}"
        )

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size

        st.write(
            f"Showing rows {start_idx + 1} to {min(end_idx, len(filtered_data))} of {len(filtered_data)}"
        )

        if st.button("Export to CSV", key=f"export_{title}"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="titanic_data.csv",
                mime="text/csv",
            )


class PlotComponent:
    def render(self, plot_path: Path, title: str = "Plot") -> None:
        st.subheader(title)

        try:
            st.image(str(plot_path), width="stretch")
        except Exception as e:
            st.error(f"Error loading plot: {e}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Refresh", key=f"refresh_{title}"):
                st.rerun()

        with col2:
            if st.button("Download", key=f"download_{title}"):
                with open(plot_path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name=plot_path.name,
                        mime="image/png",
                    )


class ModelMetricsComponent:
    def render(self, metrics: Dict[str, float], confusion_matrix: np.ndarray) -> None:
        st.subheader("Model Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")

        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")

        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")

        st.subheader("Confusion Matrix")
        self._render_confusion_matrix(confusion_matrix)

        st.subheader("Detailed Metrics")
        self._render_detailed_metrics(metrics)

    def _render_confusion_matrix(self, cm: np.ndarray) -> None:
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Survived", "Survived"],
            y=["Not Survived", "Survived"],
            color_continuous_scale="Blues",
            aspect="auto",
        )

        for i in range(len(cm)):
            for j in range(len(cm)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="red" if cm[i, j] > cm.max() / 2 else "black"),
                )

        st.plotly_chart(fig, width="stretch")

    def _render_detailed_metrics(self, metrics: Dict[str, float]) -> None:
        detailed_metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Support"],
            "Value": [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("support", 0),
            ],
        }

        st.dataframe(detailed_metrics, width="stretch")


class PredictionFormComponent:
    def render(self) -> Passenger:
        st.subheader("Passenger Information")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
                sex = st.selectbox("Gender", ["male", "female"], index=0)
                age = st.slider("Age", 0.0, 100.0, 30.0, 0.5)

            with col2:
                sibsp = st.number_input(
                    "Siblings/Spouses", min_value=0, max_value=10, value=0
                )
                parch = st.number_input(
                    "Parents/Children", min_value=0, max_value=10, value=0
                )
                fare = st.number_input(
                    "Fare", min_value=0.0, max_value=600.0, value=50.0, step=1.0
                )

            embarked = st.selectbox("Embarkation Port", ["C", "Q", "S"], index=2)

            submitted = st.form_submit_button("Predict Survival")

            if submitted:
                return Passenger(
                    passenger_id=0,
                    pclass=pclass,
                    name="User Passenger",
                    sex=sex,
                    age=age,
                    sibsp=sibsp,
                    parch=parch,
                    ticket="Ticket",
                    fare=fare,
                    cabin=None,
                    embarked=embarked,
                    survived=None,
                    title=None,
                )

        return None


class HomePage:
    def render(self, state) -> None:
        st.title("🚢 Titanic Survival Predictor")
        st.markdown("---")

        st.header("Welcome to the Titanic Machine Learning Project")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
            ### Project Overview
            This application demonstrates a complete Machine Learning pipeline for predicting
            passenger survival on the Titanic using logistic regression with gradient descent.

            ### Key Features:
            - **Exploratory Data Analysis**: Interactive visualization of Titanic dataset
            - **Model Training**: Custom implementation of logistic regression
            - **Real-time Predictions**: Interactive survival probability calculator
            - **Model Evaluation**: Comprehensive performance metrics and analysis
            """
            )

        with col2:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg",
                caption="RMS Titanic",
                width="stretch",
            )

        st.markdown("---")

        st.header("Technical Stack")

        tech_cols = st.columns(4)

        with tech_cols[0]:
            st.subheader("🛠️ Core ML")
            st.markdown(
                """
            - Custom Logistic Regression
            - Gradient Descent
            - NumPy/Pandas
            """
            )

        with tech_cols[1]:
            st.subheader("📊 Visualization")
            st.markdown(
                """
            - Matplotlib/Seaborn
            - Plotly
            - Streamlit
            """
            )

        with tech_cols[2]:
            st.subheader("🏗️ Architecture")
            st.markdown(
                """
            - Clean Architecture
            - SOLID Principles
            - Repository Pattern
            """
            )

        with tech_cols[3]:
            st.subheader("📈 Analysis")
            st.markdown(
                """
            - EDA Visualizations
            - Model Metrics
            - Statistical Analysis
            """
            )

        st.markdown("---")

        st.header("Getting Started")

        steps = st.columns(3)

        with steps[0]:
            st.subheader("1. Data Analysis")
            st.markdown(
                "Explore the Titanic dataset with interactive visualizations and statistical summaries."
            )

        with steps[1]:
            st.subheader("2. Model Training")
            st.markdown(
                "Train a logistic regression model with custom hyperparameters and monitor the training process."
            )

        with steps[2]:
            st.subheader("3. Predictions")
            st.markdown(
                "Use the trained model to predict survival probability for new passengers."
            )


class ModelExplanationPage:
    def render(self, state: AppState) -> None:
        st.title("🔍 Model Explanation & Analysis")

        if state.get("trained_model") is None:
            st.warning("Please train a model first from the Model Training page.")
            return

        model = state["trained_model"]
        dataset = state["dataset"]

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Feature Importance",
                "Prediction Analysis",
                "Model Insights",
                "Advanced Analytics",
            ]
        )

        with tab1:
            self._render_feature_importance(state, model, dataset)

        with tab2:
            self._render_prediction_analysis(state)

        with tab3:
            self._render_model_insights(state, model, dataset)

        with tab4:
            self._render_advanced_analytics(state, model, dataset)

    def _render_feature_importance(
        self, state: AppState, model: TrainedModel, dataset: Dataset
    ):
        st.subheader("Feature Importance Analysis")

        importance_data = model.get_feature_importance()
        sorted_importance = dict(
            sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                x=list(sorted_importance.values())[:15],
                y=list(sorted_importance.keys())[:15],
                orientation="h",
                title="Top 15 Most Important Features",
                labels={"x": "Importance (%)", "y": "Features"},
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Feature Statistics")
            st.metric("Total Features", len(sorted_importance))

            top_5_importance = sum(list(sorted_importance.values())[:5])
            st.metric("Top 5 Features Contribution", f"{top_5_importance:.1f}%")

            st.write("**Top 5 Features:**")
            for i, (feature, importance) in enumerate(
                list(sorted_importance.items())[:5]
            ):
                st.write(f"{i + 1}. {feature}: {importance:.2f}%")

    def _render_prediction_analysis(self, state: AppState):
        st.subheader("Individual Prediction Analysis")

        if not state.get("current_predictions"):
            st.info("Make some predictions first to see detailed analysis.")
            return

        latest_pred = state["current_predictions"][-1]
        model = state["trained_model"]
        preprocessor = state["preprocessing_pipeline"]

        prediction_service = ServiceFactory.create_prediction_service(
            model, preprocessor
        )
        explanation_service = ServiceFactory.create_explanation_service(
            prediction_service
        )

        explanation = explanation_service.explain_prediction(latest_pred.passenger)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Details")
            st.metric("Survival Probability", f"{explanation.probability:.1%}")
            st.metric(
                "Final Prediction",
                "Survived" if explanation.prediction else "Did Not Survive",
            )
            st.metric("Confidence Level", explanation.confidence_level)

            passenger = latest_pred.passenger
            st.write("**Passenger Features:**")
            feature_data = {
                "Class": passenger.pclass,
                "Gender": passenger.sex,
                "Age": passenger.age,
                "Siblings/Spouses": passenger.sibsp,
                "Parents/Children": passenger.parch,
                "Fare": f"${passenger.fare:.2f}",
                "Embarked": passenger.embarked,
            }

            for key, value in feature_data.items():
                st.write(f"- **{key}:** {value}")

        with col2:
            st.subheader("Decision Factors")
            for factor in explanation.decision_factors:
                st.write(f"• {factor}")

            st.subheader("Top Feature Impacts")
            impact_df = pd.DataFrame(
                [
                    {
                        "feature": impact.feature_name,
                        "impact": impact.impact_score,
                        "contribution": f"{impact.contribution:.1%}",
                    }
                    for impact in explanation.feature_impacts[:10]
                ]
            )

            fig = px.bar(
                impact_df,
                x="impact",
                y="feature",
                orientation="h",
                title="Top 10 Features Influencing This Prediction",
                color=impact_df["impact"] > 0,
                color_discrete_map={True: "green", False: "red"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    def _render_model_insights(
        self, state: AppState, model: TrainedModel, dataset: Dataset
    ):
        st.subheader("Model Performance Insights")

        if state.get("training_result"):
            training_result = state["training_result"]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Training Statistics:**")
                st.write(f"- Final Loss: {training_result.final_loss:.6f}")
                st.write(f"- Training Time: {training_result.training_time:.2f}s")
                st.write(f"- Learning Rate: {training_result.config.learning_rate}")
                st.write(f"- Epochs: {training_result.config.epochs}")

            with col2:
                st.write("**Model Configuration:**")
                st.write(f"- Regularization: {training_result.config.lambda_reg}")
                st.write(f"- Convergence Tol: {training_result.config.convergence_tol}")
                st.write("- Optimizer: Adam")

        st.subheader("Decision Boundary Analysis")

        feature1 = st.selectbox(
            "First Feature", options=dataset.feature_names[:10], index=0, key="feature1"
        )
        feature2 = st.selectbox(
            "Second Feature",
            options=dataset.feature_names[:10],
            index=1,
            key="feature2",
        )

        if st.button("Generate 2D Analysis"):
            self._generate_2d_analysis(feature1, feature2, model, dataset)

    def _generate_2d_analysis(
        self, feature1: str, feature2: str, model: TrainedModel, dataset: Dataset
    ):
        try:
            fig = px.scatter(
                dataset.features.assign(Survived=dataset.target),
                x=feature1,
                y=feature2,
                color="Survived",
                title=f"Decision Pattern: {feature1} vs {feature2}",
                color_discrete_map={0: "red", 1: "green"},
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Could not generate 2D analysis: {e}")

    def _render_advanced_analytics(
        self, state: AppState, model: TrainedModel, dataset: Dataset
    ):
        st.subheader("Advanced Model Analytics")

        preprocessor = state["preprocessing_pipeline"]
        prediction_service = ServiceFactory.create_prediction_service(
            model, preprocessor
        )
        explanation_service = ServiceFactory.create_explanation_service(
            prediction_service
        )

        if st.button("Run Model Diagnostics"):
            with st.spinner("Running comprehensive diagnostics..."):
                model_stats = explanation_service.get_model_statistics(model)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Model Complexity", model_stats["total_features"])
                    st.metric(
                        "Weight Magnitude", f"{model_stats['weight_magnitude']:.4f}"
                    )

                with col2:
                    st.metric("Positive Weights", model_stats["positive_weights"])
                    st.metric("Negative Weights", model_stats["negative_weights"])

                with col3:
                    weight_range = f"{model_stats['weight_range']['min']:.3f} to {model_stats['weight_range']['max']:.3f}"
                    st.metric("Weight Range", weight_range)
                    st.metric("Bias Term", f"{model_stats['bias']:.4f}")

                st.subheader("Weight Distribution")
                fig = px.histogram(
                    x=model.weights, nbins=50, title="Distribution of Model Weights"
                )
                st.plotly_chart(fig, width="stretch")


class DataAnalysisPage:
    def render(self, state: AppState) -> None:
        st.title("📊 Data Analysis")
        st.markdown("---")

        app_config = state.get("app_config", {})

        if state.get("dataset") is None:
            self._render_data_loading(state, app_config)
        else:
            self._render_data_analysis(state)

    def _render_data_loading(self, state: AppState, app_config: dict) -> None:
        st.header("Load Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Use Demo Data")

            data_path = app_config.get("ml_pipeline", {}).get(
                "data_path", "datasets/TitanicDataset.csv"
            )

            st.write(f"Using data path: `{data_path}`")

            if st.button("Load Titanic Dataset", type="primary"):
                with st.spinner("Loading dataset..."):
                    try:
                        repo = CSVDataRepository(data_path, target_column="Survived")
                        state["dataset"] = repo.load_data()

                        dataset = state["dataset"]
                        if dataset.target is not None:
                            dataset.target = dataset.target.astype(np.float64)

                        st.success("Dataset loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset from {data_path}: {e}")
                        fallback_path = "datasets/TitanicDataset.csv"
                        if data_path != fallback_path:
                            st.info(f"Trying fallback dataset: {fallback_path}")
                            try:
                                repo = CSVDataRepository(
                                    fallback_path, target_column="Survived"
                                )
                                state["dataset"] = repo.load_data()
                                st.success("Fallback dataset loaded successfully!")
                                st.rerun()
                            except Exception as fallback_e:
                                st.error(f"Fallback also failed: {fallback_e}")

        with col2:
            st.subheader("Upload Custom Data")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} rows with {len(df.columns)} columns")

                    target_col = st.selectbox("Select target column", df.columns)

                    if st.button("Process Uploaded Data"):
                        state["dataset"] = Dataset(
                            features=df.drop(columns=[target_col]),
                            target=df[target_col],
                            feature_names=list(df.drop(columns=[target_col]).columns),
                            target_name=target_col,
                        )
                        st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {e}")

    def _create_interactive_plots(self, dataset: Dataset):
        st.subheader("Интерактивный анализ выживаемости")

        x_axis = st.selectbox("Ось X", ["Age", "Fare", "Pclass"])
        y_axis = st.selectbox("Ось Y", ["Fare", "Age", "SibSp"])
        color_by = st.selectbox("Цвет по", ["Survived", "Pclass", "Sex"])

        fig = px.scatter(
            dataset.features.assign(Survived=dataset.target),
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=["Name"],
            title=f"{y_axis} vs {x_axis} по {color_by}",
        )
        st.plotly_chart(fig)

    def _render_data_analysis(self, state: AppState) -> None:
        dataset = state["dataset"]

        st.header("Dataset Overview")

        table_component = DataTableComponent()
        table_component.render(dataset.features, "Titanic Dataset")

        st.header("Statistical Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Numerical Features")
            st.dataframe(dataset.features.describe(), width="stretch")

        with col2:
            st.subheader("Categorical Features")
            categorical_stats = {}
            for col in dataset.features.select_dtypes(include=["object"]).columns:
                categorical_stats[col] = dataset.features[col].value_counts().head()
            st.json(categorical_stats)

        st.header("Data Visualizations")

        self._create_interactive_plots(dataset=dataset)

        if st.button("Generate Analysis Plots", type="primary"):
            with st.spinner("Creating visualizations..."):
                try:
                    visualizer = EDAVisualizer()

                    survival_plots = visualizer.create_survival_analysis_plots(dataset)
                    demographic_plots = visualizer.create_demographic_plots(dataset)
                    correlation_plots = visualizer.create_correlation_analysis(dataset)

                    plot_component = PlotComponent()

                    st.subheader("Survival Analysis")
                    cols = st.columns(2)
                    for i, (name, path) in enumerate(survival_plots.items()):
                        with cols[i % 2]:
                            plot_component.render(path, name.replace("_", " ").title())

                    st.subheader("Demographic Analysis")
                    cols = st.columns(2)
                    for i, (name, path) in enumerate(demographic_plots.items()):
                        with cols[i % 2]:
                            plot_component.render(path, name.replace("_", " ").title())

                    st.subheader("Correlation Analysis")
                    for name, path in correlation_plots.items():
                        plot_component.render(path, name.replace("_", " ").title())

                except Exception as e:
                    st.error(f"Error generating plots: {e}")


class ModelTrainingPage:
    def render(self, state: AppState) -> None:
        st.title("🤖 Model Training")
        st.markdown("---")

        if state.get("dataset") is None:
            st.warning("Please load a dataset first from the Data Analysis page.")
            return

        st.header("Training Configuration")

        training_config = self._render_training_controls()

        if st.button("Train Model", type="primary"):
            self._train_model(state, training_config)

        if state.get("trained_model") is not None:
            self._render_training_results(state)

    def _render_training_controls(self) -> Dict[str, Any]:
        col1, col2, col3 = st.columns(3)

        with col1:
            learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.05, 0.001)
            epochs = st.number_input("Epochs", 1000, 100000, 3000, 100)

        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)

        with col3:
            convergence_tol = st.number_input(
                "Convergence Tolerance",
                1e-8,
                1e-4,
                1e-4,
                1e-8,
            )
            batch_size = st.selectbox("Batch Size", ["full", 32, 64, 128], index=0)

        return {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "test_size": test_size,
            "random_state": random_state,
            "convergence_tol": convergence_tol,
            "batch_size": batch_size,
        }

    def _train_model(self, state: AppState, config: Dict[str, Any]) -> None:
        try:
            with st.spinner("Training model..."):
                if state.get("preprocessing_pipeline") is None:
                    state["preprocessing_pipeline"] = (
                        PreprocessorFactory.create_titanic_preprocessor()
                    )

                preprocessor = state["preprocessing_pipeline"]
                training_service = ServiceFactory.create_training_service(preprocessor)

                training_config = TrainingConfig(
                    learning_rate=config["learning_rate"],
                    epochs=config["epochs"],
                    test_size=config["test_size"],
                    random_state=config["random_state"],
                    convergence_tol=config["convergence_tol"],
                )

                training_result = training_service.train_model(
                    state["dataset"], training_config
                )
                state["trained_model"] = training_result.model
                state["training_history"] = training_result.learning_curve
                state["training_result"] = training_result

                state["preprocessing_artifacts"] = (
                    training_result.model.preprocessing_artifacts
                )

                if hasattr(training_service, "poly_transformer"):
                    state["trained_model"].preprocessing_artifacts = {
                        "poly_transformer": training_service.poly_transformer,
                        "X_mean": getattr(training_service, "X_mean", None),
                        "X_std": getattr(training_service, "X_std", None),
                    }

                st.success("Model trained successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"Error training model: {e}")

    def _render_training_results(self, state: AppState) -> None:
        st.header("Training Results")

        model = state["trained_model"]
        training_result = state.get("training_result")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Information")
            st.metric("Number of Features", len(model.feature_names))
            st.metric("Final Loss", f"{state['training_history'][-1]:.6f}")

            if training_result:
                st.metric("Training Time", f"{training_result.training_time:.2f}s")
            else:
                st.metric("Training Time", "N/A")

        with col2:
            st.subheader("Feature Importance")
            importance_data = {
                "Feature": model.feature_names,
                "Importance": np.abs(model.weights),
            }
            importance_df = pd.DataFrame(importance_data).sort_values(
                "Importance", ascending=False
            )
            st.dataframe(importance_df.head(55), width="stretch")

        st.subheader("Training Progress")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(state["training_history"]))),
                y=state["training_history"],
                mode="lines",
                name="Training Loss",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            showlegend=True,
            template="plotly_white",
        )

        st.plotly_chart(fig, width="stretch")

        if st.button("Evaluate Model", type="secondary"):
            self._evaluate_model(state)

    def _evaluate_model(self, state: AppState) -> None:
        try:
            with st.spinner("Evaluating model..."):
                preprocessor = state["preprocessing_pipeline"]
                training_service = ServiceFactory.create_training_service(preprocessor)

                evaluation_result = training_service.evaluate_model(
                    state["trained_model"], state["dataset"]
                )

                metrics_component = ModelMetricsComponent()
                metrics_component.render(
                    evaluation_result.classification_report,
                    evaluation_result.confusion_matrix,
                )

        except Exception as e:
            st.error(f"Error evaluating model: {e}")


class PredictionPage:
    def render(self, state: AppState) -> None:
        st.title("🔮 Survival Prediction")
        st.markdown("---")

        if state.get("trained_model") is None:
            st.warning("Please train a model first from the Model Training page.")
            return

        with st.expander("Model Debug Info"):
            model = state["trained_model"]
            st.write(f"Number of features in model: {len(model.feature_names)}")
            st.write(f"Feature names: {model.feature_names}")
            st.write(f"Model weights shape: {model.weights.shape}")
            st.write(f"Model bias: {model.bias}")

        st.header("Make a Prediction")

        form_component = PredictionFormComponent()
        passenger = form_component.render()

        if passenger is not None:
            self._make_prediction(state, passenger)

        if state.get("current_predictions"):
            self._render_prediction_history(state)

    def _make_prediction(self, state: AppState, passenger: Passenger) -> None:
        try:
            with st.spinner("Calculating prediction..."):
                preprocessor = state["preprocessing_pipeline"]
                prediction_service = ServiceFactory.create_prediction_service(
                    state["trained_model"], preprocessor
                )

                prediction_result = prediction_service.predict_survival(passenger)

                if "current_predictions" not in state:
                    state["current_predictions"] = []

                state["current_predictions"].append(prediction_result)

                with st.expander("Prediction Debug Info"):
                    st.write(f"Probability: {prediction_result.probability:.4f}")
                    st.write(f"Prediction: {prediction_result.prediction}")
                    st.write(f"Confidence: {prediction_result.confidence:.4f}")
                    st.write(f"Passenger features: {passenger}")

                self._render_prediction_result(prediction_result)

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error(f"Detailed error: {str(e)}")

    def _render_prediction_result(self, prediction: Any) -> None:
        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            probability = prediction.probability
            survived = prediction.prediction

            st.metric(
                "Survival Probability",
                f"{probability:.1%}",
                delta="Survived" if survived else "Did Not Survive",
                delta_color="normal" if survived else "inverse",
            )

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Survival Chance"},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "lightcoral"},
                            {"range": [30, 70], "color": "lightyellow"},
                            {"range": [70, 100], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )

            st.plotly_chart(gauge_fig, width="stretch")

        with col2:
            st.subheader("Passenger Details")
            passenger_info = {
                "Class": prediction.passenger.pclass,
                "Gender": prediction.passenger.sex,
                "Age": prediction.passenger.age,
                "Siblings/Spouses": prediction.passenger.sibsp,
                "Parents/Children": prediction.passenger.parch,
                "Fare": f"${prediction.passenger.fare:.2f}",
                "Embarkation": prediction.passenger.embarked,
            }

            for key, value in passenger_info.items():
                st.write(f"**{key}:** {value}")

    def _render_prediction_history(self, state: AppState) -> None:
        st.header("Prediction History")

        history_data = []
        for pred in state["current_predictions"][-10:]:
            history_data.append(
                {
                    "Probability": f"{pred.probability:.1%}",
                    "Prediction": "Survived" if pred.prediction else "Did Not Survive",
                    "Age": pred.passenger.age,
                    "Class": pred.passenger.pclass,
                    "Gender": pred.passenger.sex,
                    "Timestamp": pred.timestamp.strftime("%H:%M:%S"),
                }
            )

        if history_data:
            st.dataframe(pd.DataFrame(history_data), width="stretch")


class TitanicApp:
    def __init__(self, app_config: dict = None):
        self.pages = {
            "Home": HomePage(),
            "Data Analysis": DataAnalysisPage(),
            "Model Training": ModelTrainingPage(),
            "Model Explanation": ModelExplanationPage(),
            "Predictions": PredictionPage(),
        }
        self.app_config = app_config or {}

    def run(self) -> None:
        st.set_page_config(
            page_title="Titanic Survival Predictor",
            page_icon="🚢",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        if "app_state" not in st.session_state:
            st.session_state.app_state = {
                "current_page": "Home",
                "dataset": None,
                "trained_model": None,
                "preprocessing_pipeline": None,
                "current_predictions": [],
                "training_history": [],
                "app_config": self.app_config,  # Передаем конфиг в состояние
            }

        self._render_sidebar()
        self._render_current_page()

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.title("🚢 Titanic ML")
            st.markdown("---")

            selected_page = st.radio(
                "Navigation",
                list(self.pages.keys()),
                index=list(self.pages.keys()).index(
                    st.session_state.app_state["current_page"]
                ),
            )

            st.session_state.app_state["current_page"] = selected_page

            st.markdown("---")

            st.subheader("App Status")

            app_config = st.session_state.app_state.get("app_config", {})
            data_path = app_config.get("ml_pipeline", {}).get(
                "data_path", "datasets/TitanicDataset.csv"
            )

            dataset_status = (
                "✅ Loaded"
                if st.session_state.app_state.get("dataset")
                else "❌ Not Loaded"
            )
            model_status = (
                "✅ Trained"
                if st.session_state.app_state.get("trained_model")
                else "❌ Not Trained"
            )

            st.write(f"Dataset: {dataset_status}")
            st.write(f"Model: {model_status}")
            st.write(f"Data path: {data_path}")

            if st.session_state.app_state.get("dataset"):
                dataset = st.session_state.app_state["dataset"]
                st.write(f"Rows: {len(dataset.features)}")
                st.write(f"Features: {len(dataset.feature_names)}")

            st.markdown("---")

            if st.button("Clear All Data", type="secondary"):
                st.session_state.app_state = {
                    "current_page": "Home",
                    "dataset": None,
                    "trained_model": None,
                    "preprocessing_pipeline": None,
                    "current_predictions": [],
                    "training_history": [],
                    "training_result": None,
                    "app_config": self.app_config,
                }
                st.rerun()

    def _render_current_page(self) -> None:
        current_page = st.session_state.app_state["current_page"]
        page_instance = self.pages[current_page]
        page_instance.render(st.session_state.app_state)
