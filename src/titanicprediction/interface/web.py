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
        self, data: pd.DataFrame, title: str = "Таблица данных", page_size: int = 10
    ) -> None:
        st.subheader(title)

        col1, col2 = st.columns([2, 1])

        with col1:
            search_term = st.text_input("Поиск...", key=f"search_{title}")

        with col2:
            sort_column = st.selectbox("Сортировка", data.columns, key=f"sort_{title}")

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
            "Страница", min_value=1, max_value=total_pages, value=1, key=f"page_{title}"
        )

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size

        st.write(
            f"Показаны строки с {start_idx + 1} до {min(end_idx, len(filtered_data))} из {len(filtered_data)}"
        )

        if st.button("Экспорт в CSV", key=f"export_{title}"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name="titanic_data.csv",
                mime="text/csv",
            )


class PlotComponent:
    def render(self, plot_path: Path, title: str = "График") -> None:
        st.subheader(title)

        try:
            st.image(str(plot_path), width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки графика: {e}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Обновить", key=f"refresh_{title}"):
                st.rerun()

        with col2:
            if st.button("Скачать", key=f"download_{title}"):
                with open(plot_path, "rb") as file:
                    st.download_button(
                        label="Скачать изображение",
                        data=file,
                        file_name=plot_path.name,
                        mime="image/png",
                    )


class ModelMetricsComponent:
    def render(self, metrics: Dict[str, float], confusion_matrix: np.ndarray) -> None:
        st.subheader("Метрики производительности модели")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy / Точность", f"{metrics.get('accuracy', 0):.3f}")

        with col2:
            st.metric(
                "Precision / Точность верно положительных",
                f"{metrics.get('precision', 0):.3f}",
            )

        with col3:
            st.metric("Recall / Полнота", f"{metrics.get('recall', 0):.3f}")

        with col4:
            st.metric("F1-Score / F1-мера", f"{metrics.get('f1_score', 0):.3f}")

        st.subheader("Матрица неточностей")
        self._render_confusion_matrix(confusion_matrix)

        st.subheader("Подробные показатели")
        self._render_detailed_metrics(metrics)

    def _render_confusion_matrix(self, cm: np.ndarray) -> None:
        fig = px.imshow(
            cm,
            labels=dict(x="Предсказано", y="Актуально", color="Число"),
            x=["Не выжило", "Выжило"],
            y=["Не выжило", "Выжило"],
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
            "Метрики": ["Accuracy", "Precision", "Recall", "F1-Score", "Support"],
            "Значение": [
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
        st.subheader("Информация о пассажирах")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                pclass = st.selectbox("Класс пассажира", [1, 2, 3], index=0)
                sex = st.selectbox("Пол", ["male", "female"], index=0)
                age = st.slider("Возраст", 0.0, 100.0, 30.0, 0.5)

            with col2:
                sibsp = st.number_input(
                    "Братья и сестры/Супруги", min_value=0, max_value=10, value=0
                )
                parch = st.number_input(
                    "Родители/Дети", min_value=0, max_value=10, value=0
                )
                fare = st.number_input(
                    "Плата", min_value=0.0, max_value=600.0, value=50.0, step=1.0
                )

            embarked = st.selectbox("Порт посадки", ["C", "Q", "S"], index=2)

            submitted = st.form_submit_button("Предсказать шанс выживания")

            if submitted:
                return Passenger(
                    passenger_id=0,
                    pclass=pclass,
                    name="Пользователь",
                    sex=sex,
                    age=age,
                    sibsp=sibsp,
                    parch=parch,
                    ticket="Билет",
                    fare=fare,
                    cabin=None,
                    embarked=embarked,
                    survived=None,
                    title=None,
                )

        return None


class HomePage:
    def render(self, state) -> None:
        st.title("🚢 Модель данных выживания на титанике")
        st.markdown("---")

        st.header("Проект по анализу данных и машинному обучению")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
### О проекте
В современном мире с динамическим развитием AI-сферы мало людей, знакомых с технической базой по ИИ-продуктам. А также общество разделилось на две крайности, первые гласят что ИИ заменит всех нас, программисты уйдут на завод, а вторая половина говорит что ИИ это всемирный пузырь и бизнес. И те, и те правы, и не правы одновременно.
В искуственном интеллекте есть несколько компонентов. Самая банальная — нейронная сеть (самая простейшая называется персептрон, к примеру), о том как их создавать есть множество проектов и документации. Но в сфере школьных проектов мало кто говорить о другой важнейшей части — Machine Learning, машинном обучении.
Именно этой части и посвящен мой проект. Я планирую создать простейшую модель, которая на основе существующего набора данных по смертности на «Титанике» предсказать шанс выживания пользователя с персонализацией по классу билета, цены билета, полу, и возрасту. Это затронет как и знания из области науки о данных (Data Science) так и знания математического анализа (в виде производной и градиентского спуска).
При выполнении проекта я буду следовать следующему плану:
1. Загрузка и изучение данных.
2. Проведение разведочного анализа для нахождения закономерностей и аномалий.
3. Подготовка данных — обработка пропусков, кодирование категорий.
4. Реализование модели — через логистическую регрессию с градиентным спуском.
5. Обучение и оценка точности модели.
Из этого можно сформировать следующие характеристики проекта:
Цель: Изучить ML и построить свою модель.
Проблема: В современном мире с динамическим развитием AI-сферы мало людей, знакомых с технической базой по ИИ-продуктам.
Актуальность: Активно развивается искуственный интеллект, а также профессия ML-инженера — одна из самых перспективных среди направлений в программировании и аналитике данных.
Сфомируем задачи:
1. Изучить машинное обучение;
2. Изучить основы Data Science и аналитики;
3. Изучить призводную, градиентский спуск и линейную регрессию;
4. Изучить методы классификации модели;
5. Провести разведочный анализ данных;
6. Обучить модель и оценить ее;
7. Реализовать клиентский интерфейс для взаимодействия с моделью.
            """
            )

        with col2:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg",
                caption="Титаник",
                width="stretch",
            )

        st.markdown("---")

        st.header("Технический стек")

        tech_cols = st.columns(4)

        with tech_cols[0]:
            st.subheader("🛠️ Машинное обучение")
            st.markdown(
                """
            - Кастомная логистическая регрессия
            - Градиентный спуск
            - NumPy/Pandas
            """
            )

        with tech_cols[1]:
            st.subheader("📊 Визуализация")
            st.markdown(
                """
            - Matplotlib/Seaborn
            - Plotly
            - Streamlit
            """
            )

        with tech_cols[2]:
            st.subheader("🏗️ Архитектура")
            st.markdown(
                """
            - Чистая архитектура
            - SOLID
            - Паттерны ООП
            - ACID
            - DRY, KISS
            """
            )

        with tech_cols[3]:
            st.subheader("📈 Аналитика")
            st.markdown(
                """
            - Визуализация EDA
            - Метрики модели
            - Статистический анализ
            - Анализ решений модели
            """
            )

        st.markdown("---")

        st.header("Как начать")

        steps = st.columns(3)

        with steps[0]:
            st.subheader("1. Анализ данных")
            st.markdown(
                "Изучение датасета Титаника с интерактивными графиками и статистическими выводами."
            )

        with steps[1]:
            st.subheader("2. Обучение модели")
            st.markdown(
                "Тренировка и обучение модели по логистической регрессии, а также объяснение и мониторинг модели."
            )

        with steps[2]:
            st.subheader("3. Предсказание")
            st.markdown(
                "Использование обученной модели длля предсказания шанса выживания для пассажира."
            )


class ModelExplanationPage:
    def render(self, state: AppState) -> None:
        st.title("🔍 Объяснение модели и анализ")

        if state.get("trained_model") is None:
            st.warning("Пожалуйста обучите модель перед анализом.")
            return

        model = state["trained_model"]
        dataset = state["dataset"]

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Важность фич",
                "Анализ предсказания",
                "Инсайты модели",
                "Продвинутая аналитика",
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
        st.subheader("Важность фич")

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
                title="Топ 15 наиболее важных фич",
                labels={"x": "Важность (%)", "y": "Фичи"},
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Статистика фич")
            st.metric("Всего фич", len(sorted_importance))

            top_5_importance = sum(list(sorted_importance.values())[:5])
            st.metric("Топ 5 фич по вкладу", f"{top_5_importance:.1f}%")

            st.write("**Топ 5 фич:**")
            for i, (feature, importance) in enumerate(
                list(sorted_importance.items())[:5]
            ):
                st.write(f"{i + 1}. {feature}: {importance:.2f}%")

    def _render_prediction_analysis(self, state: AppState):
        st.subheader("Индивидуальная аналитика предсказания")

        if not state.get("current_predictions"):
            st.info("Сделайте какие нибудь предсказания перед началом.")
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
            st.subheader("Детали предсказания")
            st.metric("Шанс выживания", f"{explanation.probability:.1%}")
            st.metric(
                "Финальное решение",
                "Выжил" if explanation.prediction else "Не выжил",
            )
            st.metric("Уровень уверенности", explanation.confidence_level)

            passenger = latest_pred.passenger
            st.write("**Фичи пассажира:**")
            feature_data = {
                "Класс": passenger.pclass,
                "Пол": passenger.sex,
                "Возраст": passenger.age,
                "Братья и сестры": passenger.sibsp,
                "Родители/дети": passenger.parch,
                "Сумма": f"${passenger.fare:.2f}",
                "Погруженный порт": passenger.embarked,
            }

            for key, value in feature_data.items():
                st.write(f"- **{key}:** {value}")

        with col2:
            st.subheader("Факторы, влияющие на принятие решения")
            for factor in explanation.decision_factors:
                st.write(f"• {factor}")

            st.subheader("Топ факторов, влияющих на принятие решение")
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
                title="Топ 10 фич влияющие на это решение",
                color=impact_df["impact"] > 0,
                color_discrete_map={True: "green", False: "red"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    def _render_model_insights(
        self, state: AppState, model: TrainedModel, dataset: Dataset
    ):
        st.subheader("Инсайты модели")

        if state.get("training_result"):
            training_result = state["training_result"]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Статистика обучения:**")
                st.write(f"- Финальные потери: {training_result.final_loss:.6f}")
                st.write(f"- Время обучения: {training_result.training_time:.2f}s")
                st.write(f"- Оценка обучения: {training_result.config.learning_rate}")
                st.write(f"- Эпохи: {training_result.config.epochs}")

            with col2:
                st.write("**Настройки модели:**")
                st.write(f"- Регуляризация: {training_result.config.lambda_reg}")
                st.write(f"- Convergence Tol: {training_result.config.convergence_tol}")
                st.write("- Оптимизатор: Adam")

        st.subheader("Анализ границ решений")

        feature1 = st.selectbox(
            "Первая фича", options=dataset.feature_names[:10], index=0, key="feature1"
        )
        feature2 = st.selectbox(
            "Вторая фича",
            options=dataset.feature_names[:10],
            index=1,
            key="feature2",
        )

        if st.button("Сгенерировать 2D график"):
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
        st.subheader("Продвинутая аналитика модели")

        preprocessor = state["preprocessing_pipeline"]
        prediction_service = ServiceFactory.create_prediction_service(
            model, preprocessor
        )
        explanation_service = ServiceFactory.create_explanation_service(
            prediction_service
        )

        if st.button("Запустить диагностику модели"):
            with st.spinner("Запуск комплексной диагностики..."):
                model_stats = explanation_service.get_model_statistics(model)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Сложность модели", model_stats["total_features"])
                    st.metric(
                        "Величина весов", f"{model_stats['weight_magnitude']:.4f}"
                    )

                with col2:
                    st.metric("Положительные веса", model_stats["positive_weights"])
                    st.metric("Отрицательные веса", model_stats["negative_weights"])

                with col3:
                    weight_range = f"{model_stats['weight_range']['min']:.3f} to {model_stats['weight_range']['max']:.3f}"
                    st.metric("Диапазон весов", weight_range)
                    st.metric("Смещение", f"{model_stats['bias']:.4f}")

                st.subheader("Распределение весов")
                fig = px.histogram(
                    x=model.weights, nbins=50, title="Distribution of Model Weights"
                )
                st.plotly_chart(fig, width="stretch")


class DataAnalysisPage:
    def render(self, state: AppState) -> None:
        st.title("📊 Анализ данных")
        st.markdown("---")

        app_config = state.get("app_config", {})

        if state.get("dataset") is None:
            self._render_data_loading(state, app_config)
        else:
            self._render_data_analysis(state)

    def _render_data_loading(self, state: AppState, app_config: dict) -> None:
        st.header("Загрузить датасет")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Использовать демо данные")

            data_path = app_config.get("ml_pipeline", {}).get(
                "data_path", "datasets/TitanicDataset.csv"
            )

            st.write(f"Используется путь к данным: `{data_path}`")

            if st.button("Загрузить датасет Титаника", type="primary"):
                with st.spinner("Загрузка датасета..."):
                    try:
                        repo = CSVDataRepository(data_path, target_column="Survived")
                        state["dataset"] = repo.load_data()

                        dataset = state["dataset"]
                        if dataset.target is not None:
                            dataset.target = dataset.target.astype(np.float64)

                        st.success("Датасет успешно загружен!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка загрузки датасета из {data_path}: {e}")
                        fallback_path = "datasets/TitanicDataset.csv"
                        if data_path != fallback_path:
                            st.info(f"Пробуем запасной датасет: {fallback_path}")
                            try:
                                repo = CSVDataRepository(
                                    fallback_path, target_column="Survived"
                                )
                                state["dataset"] = repo.load_data()
                                st.success("Запасной датасет успешно загружен!")
                                st.rerun()
                            except Exception as fallback_e:
                                st.error(f"Запасной также не сработал: {fallback_e}")

        with col2:
            st.subheader("Загрузить свои данные")
            uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(
                        f"Загружено {len(df)} строк с {len(df.columns)} колонками"
                    )

                    target_col = st.selectbox("Выберите целевую колонку", df.columns)

                    if st.button("Обработать загруженные данные"):
                        state["dataset"] = Dataset(
                            features=df.drop(columns=[target_col]),
                            target=df[target_col],
                            feature_names=list(df.drop(columns=[target_col]).columns),
                            target_name=target_col,
                        )
                        st.rerun()

                except Exception as e:
                    st.error(f"Ошибка обработки файла: {e}")

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

        st.header("Обзор датасета")

        table_component = DataTableComponent()
        table_component.render(dataset.features, "Датасет Титаника")

        st.header("Статистическая сводка")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Числовые характеристики")
            st.dataframe(dataset.features.describe(), width="stretch")

        with col2:
            st.subheader("Категориальные характеристики")
            categorical_stats = {}
            for col in dataset.features.select_dtypes(include=["object"]).columns:
                categorical_stats[col] = dataset.features[col].value_counts().head()
            st.json(categorical_stats)

        st.header("Визуализации данных")

        self._create_interactive_plots(dataset=dataset)

        if st.button("Сгенерировать графики анализа", type="primary"):
            with st.spinner("Создание визуализаций..."):
                try:
                    visualizer = EDAVisualizer()

                    survival_plots = visualizer.create_survival_analysis_plots(dataset)
                    demographic_plots = visualizer.create_demographic_plots(dataset)
                    correlation_plots = visualizer.create_correlation_analysis(dataset)

                    plot_component = PlotComponent()

                    st.subheader("Анализ выживаемости")
                    cols = st.columns(2)
                    for i, (name, path) in enumerate(survival_plots.items()):
                        with cols[i % 2]:
                            plot_component.render(path, name.replace("_", " ").title())

                    st.subheader("Демографический анализ")
                    cols = st.columns(2)
                    for i, (name, path) in enumerate(demographic_plots.items()):
                        with cols[i % 2]:
                            plot_component.render(path, name.replace("_", " ").title())

                    st.subheader("Корреляционный анализ")
                    for name, path in correlation_plots.items():
                        plot_component.render(path, name.replace("_", " ").title())

                except Exception as e:
                    st.error(f"Ошибка генерации графиков: {e}")


class ModelTrainingPage:
    def render(self, state: AppState) -> None:
        st.title("🤖 Обучение модели")
        st.markdown("---")

        if state.get("dataset") is None:
            st.warning(
                "Пожалуйста сначала загрузите датасет на странице Анализ данных."
            )
            return

        st.header("Конфигурация обучения")

        training_config = self._render_training_controls()

        if st.button("Обучить модель", type="primary"):
            self._train_model(state, training_config)

        if state.get("trained_model") is not None:
            self._render_training_results(state)

    def _render_training_controls(self) -> Dict[str, Any]:
        col1, col2, col3 = st.columns(3)

        with col1:
            learning_rate = st.slider("Скорость обучения", 0.001, 0.5, 0.05, 0.001)
            epochs = st.number_input("Эпохи", 1000, 100000, 3000, 100)

        with col2:
            test_size = st.slider("Размер теста", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Случайное состояние", 0, 100, 42)

        with col3:
            convergence_tol = st.number_input(
                "Допуск сходимости",
                1e-8,
                1e-4,
                1e-4,
                1e-8,
            )
            batch_size = st.selectbox("Размер батча", ["полный", 32, 64, 128], index=0)

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
            with st.spinner("Обучение модели..."):
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

                st.success("Модель успешно обучена!")
                st.rerun()

        except Exception as e:
            st.error(f"Ошибка обучения модели: {e}")

    def _render_training_results(self, state: AppState) -> None:
        st.header("Результаты обучения")

        model = state["trained_model"]
        training_result = state.get("training_result")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Информация о модели")
            st.metric("Количество фич", len(model.feature_names))
            st.metric("Финальные потери", f"{state['training_history'][-1]:.6f}")

            if training_result:
                st.metric("Время обучения", f"{training_result.training_time:.2f}s")
            else:
                st.metric("Время обучения", "Н/Д")

        with col2:
            st.subheader("Важность фич")
            importance_data = {
                "Фича": model.feature_names,
                "Важность": np.abs(model.weights),
            }
            importance_df = pd.DataFrame(importance_data).sort_values(
                "Важность", ascending=False
            )
            st.dataframe(importance_df.head(55), width="stretch")

        st.subheader("Прогресс обучения")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(state["training_history"]))),
                y=state["training_history"],
                mode="lines",
                name="Потери обучения",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Потери обучения во времени",
            xaxis_title="Эпоха",
            yaxis_title="Потери",
            showlegend=True,
            template="plotly_white",
        )

        st.plotly_chart(fig, width="stretch")

        if st.button("Оценить модель", type="secondary"):
            self._evaluate_model(state)

    def _evaluate_model(self, state: AppState) -> None:
        try:
            with st.spinner("Оценка модели..."):
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
            st.error(f"Ошибка оценки модели: {e}")


class PredictionPage:
    def render(self, state: AppState) -> None:
        st.title("🔮 Предсказание выживания")
        st.markdown("---")

        if state.get("trained_model") is None:
            st.warning("Пожалуйста сначала обучите модель на странице Обучение модели.")
            return

        with st.expander("Отладочная информация модели"):
            model = state["trained_model"]
            st.write(f"Количество фич в модели: {len(model.feature_names)}")
            st.write(f"Названия фич: {model.feature_names}")
            st.write(f"Форма весов модели: {model.weights.shape}")
            st.write(f"Смещение модели: {model.bias}")

        st.header("Сделать предсказание")

        form_component = PredictionFormComponent()
        passenger = form_component.render()

        if passenger is not None:
            self._make_prediction(state, passenger)

        if state.get("current_predictions"):
            self._render_prediction_history(state)

    def _make_prediction(self, state: AppState, passenger: Passenger) -> None:
        try:
            with st.spinner("Вычисление предсказания..."):
                preprocessor = state["preprocessing_pipeline"]
                prediction_service = ServiceFactory.create_prediction_service(
                    state["trained_model"], preprocessor
                )

                prediction_result = prediction_service.predict_survival(passenger)

                if "current_predictions" not in state:
                    state["current_predictions"] = []

                state["current_predictions"].append(prediction_result)

                with st.expander("Отладочная информация предсказания"):
                    st.write(f"Вероятность: {prediction_result.probability:.4f}")
                    st.write(f"Предсказание: {prediction_result.prediction}")
                    st.write(f"Уверенность: {prediction_result.confidence:.4f}")
                    st.write(f"Фичи пассажира: {passenger}")

                self._render_prediction_result(prediction_result)

        except Exception as e:
            st.error(f"Ошибка выполнения предсказания: {e}")
            st.error(f"Детальная ошибка: {str(e)}")

    def _render_prediction_result(self, prediction: Any) -> None:
        st.header("Результат предсказания")

        col1, col2 = st.columns(2)

        with col1:
            probability = prediction.probability
            survived = prediction.prediction

            st.metric(
                "Вероятность выживания",
                f"{probability:.1%}",
                delta="Выжил" if survived else "Не выжил",
                delta_color="normal" if survived else "inverse",
            )

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Шанс выживания"},
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
            st.subheader("Детали пассажира")
            passenger_info = {
                "Класс": prediction.passenger.pclass,
                "Пол": prediction.passenger.sex,
                "Возраст": prediction.passenger.age,
                "Братья/Супруги": prediction.passenger.sibsp,
                "Родители/Дети": prediction.passenger.parch,
                "Плата": f"${prediction.passenger.fare:.2f}",
                "Порт посадки": prediction.passenger.embarked,
            }

            for key, value in passenger_info.items():
                st.write(f"**{key}:** {value}")

    def _render_prediction_history(self, state: AppState) -> None:
        st.header("История предсказаний")

        history_data = []
        for pred in state["current_predictions"][-10:]:
            history_data.append(
                {
                    "Вероятность": f"{pred.probability:.1%}",
                    "Предсказание": "Выжил" if pred.prediction else "Не выжил",
                    "Возраст": pred.passenger.age,
                    "Класс": pred.passenger.pclass,
                    "Пол": pred.passenger.sex,
                    "Время": pred.timestamp.strftime("%H:%M:%S"),
                }
            )

        if history_data:
            st.dataframe(pd.DataFrame(history_data), width="stretch")


class TitanicApp:
    def __init__(self, app_config: dict = None):
        self.pages = {
            "Главная": HomePage(),
            "Анализ данных": DataAnalysisPage(),
            "Обучение модели": ModelTrainingPage(),
            "Объяснение модели": ModelExplanationPage(),
            "Предсказания": PredictionPage(),
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
                "current_page": "Главная",
                "dataset": None,
                "trained_model": None,
                "preprocessing_pipeline": None,
                "current_predictions": [],
                "training_history": [],
                "app_config": self.app_config,
            }

        self._render_sidebar()
        self._render_current_page()

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.title("🚢 Titanic ML")
            st.markdown("---")

            selected_page = st.radio(
                "Навигация",
                list(self.pages.keys()),
                index=list(self.pages.keys()).index(
                    st.session_state.app_state["current_page"]
                ),
            )

            st.session_state.app_state["current_page"] = selected_page

            st.markdown("---")

            st.subheader("Статус приложения")

            app_config = st.session_state.app_state.get("app_config", {})
            data_path = app_config.get("ml_pipeline", {}).get(
                "data_path", "datasets/TitanicDataset.csv"
            )

            dataset_status = (
                "✅ Загружен"
                if st.session_state.app_state.get("dataset")
                else "❌ Не загружен"
            )
            model_status = (
                "✅ Обучена"
                if st.session_state.app_state.get("trained_model")
                else "❌ Не обучена"
            )

            st.write(f"Датасет: {dataset_status}")
            st.write(f"Модель: {model_status}")
            st.write(f"Путь к данным: {data_path}")

            if st.session_state.app_state.get("dataset"):
                dataset = st.session_state.app_state["dataset"]
                st.write(f"Строки: {len(dataset.features)}")
                st.write(f"Фичи: {len(dataset.feature_names)}")

            st.markdown("---")

            if st.button("Очистить все данные", type="secondary"):
                st.session_state.app_state = {
                    "current_page": "Главная",
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
