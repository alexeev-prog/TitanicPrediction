# web.py
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

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
    dataset: Dataset | None
    trained_model: TrainedModel | None
    preprocessing_pipeline: Any | None
    current_predictions: list[Any]
    training_history: list[float]
    training_result: Any | None


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
            download_key = f"download_{title}_{hash(plot_path)}"

            try:
                with open(plot_path, "rb") as file:
                    file_data = file.read()
                    st.download_button(
                        label="Скачать изображение",
                        data=file_data,
                        file_name=plot_path.name,
                        mime="image/png",
                        key=download_key,
                    )
            except Exception as e:
                st.error(f"Ошибка при подготовке файла для скачивания: {e}")


class ModelMetricsComponent:
    def render(self, metrics: dict[str, float], confusion_matrix: np.ndarray) -> None:
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
            labels={"x": "Предсказано", "y": "Актуально", "color": "Число"},
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
                    font={"color": "red" if cm[i, j] > cm.max() / 2 else "black"},
                )

        st.plotly_chart(fig, width="stretch")

    def _render_detailed_metrics(self, metrics: dict[str, float]) -> None:
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


class DocumentationPage:
    def render(self, state: AppState) -> None:
        st.title("📚 Теоретическая документация")
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Теоретическая часть",
                "Практическая часть",
                "Результаты и выводы",
                "Источники",
                "Техническая документация",
            ]
        )

        with tab1:
            self._render_theoretical_part()

        with tab2:
            self._render_practical_part()

        with tab3:
            self._render_results_conclusions()

        with tab4:
            self._render_sources()

        with tab5:
            self._render_technical_documentation()

    def _render_theoretical_part(self):
        st.header("3. Теоретическая часть")

        st.subheader("3.1 Что такое машинное обучение?")
        st.markdown("""
        Традиционное программирование предполагает создание разработчиком четкого и детального алгоритма,
        предписывающего компьютеру точную последовательность действий для решения задачи. Однако для многих
        практических задач, таких как распознавание образов или прогнозирование, создание исчерпывающего
        набора правил оказывается чрезвычайно сложным или вовсе невозможным.

        **Машинное обучение** предлагает принципиально иной подход. В его основе лежит идея о том, что
        компьютер не получает готовых инструкций, а обучается самостоятельно находить закономерности и
        принимать решения путем анализа большого объема эмпирических данных.
        """)

        st.subheader("3.2 Типы машинного обучения")
        st.markdown("""
        В зависимости от характера доступных данных и способа взаимодействия модели со средой машинное
        обучение принято классифицировать на несколько основных типов:

        #### 1. Обучение с учителем (Supervised Learning)
        - Модель обучается на размеченных данных
        - Каждый объект содержит входные признаки и соответствующую метку
        - Задача: выявить взаимосвязь между признаками и целевой переменной
        - **Пример**: Предсказание выживаемости на Титанике

        #### 2. Обучение без учителя (Unsupervised Learning)
        - Модель работает с данными без меток
        - Задача: обнаружить скрытые структуры и закономерности
        - **Примеры**: Кластеризация, понижение размерности

        #### 3. Обучение с подкреплением (Reinforcement Learning)
        - Модель обучается через взаимодействие со средой
        - Получает награду за правильные действия
        - **Примеры**: Игровые ИИ, робототехника
        """)

        st.subheader("3.3 Задача классификации и логистическая регрессия")
        st.markdown("""
        **Задача классификации** - одна из фундаментальных проблем в области обучения с учителем.
        Её суть заключается в автоматическом отнесении объектов к одному из заранее заданных классов
        на основе анализа их признаков.

        **Логистическая регрессия** - алгоритм контролируемого обучения, используемый для решения
        задач классификации. Несмотря на название, является алгоритмом классификации, а не регрессии.

        #### Как работает логистическая регрессия:
        1. **Линейная комбинация**: Вычисляется взвешенная сумма входных признаков
        2. **Вероятностная трансформация**: Используется логистическая функция (сигмоида) для преобразования в вероятность
        3. **Принятие решения**: Вероятность сравнивается с пороговым значением (обычно 0.5)

        **Сигмоида** - гладкая монотонная возрастающая нелинейная функция, имеющая форму буквы «S».
        Математический вид: f(x) = 1 / (1 + e^-x)
        """)

        st.subheader("3.4 Математические основы: производная и градиентный спуск")
        st.markdown("""
        **Производная** - одно из самых важных понятий математического анализа.
        Характеризует скорость изменения функции в данной точке.

        **Градиент** - вектор, составленный из частных производных функции по всем её переменным.
        Указывает направление наискорейшего роста функции.

        **Градиентный спуск** - итеративный алгоритм численной оптимизации, используемый для
        нахождения локального минимума функции.

        #### Алгоритм градиентного спуска:
        1. **Инициализация**: Параметры инициализируются случайными значениями
        2. **Итеративное обновление**: На каждом шаге вычисляется градиент функции потерь
        3. **Корректировка параметров**: Параметры обновляются в направлении, противоположном градиенту

        **Скорость обучения** - гиперпараметр, определяющий размер шага на каждой итерации.
        """)

        st.subheader("3.5 Этапы работы над Data Science проектом")
        st.markdown("""
        Стандартный жизненный цикл Data Science проекта включает следующие этапы:

        1. **Формулирование проблемы**: Определение цели и типа задачи
        2. **Сбор данных**: Поиск и сбор релевантных данных
        3. **Предобработка данных**: Очистка и преобразование данных
        4. **Разведочный анализ**: Изучение данных и выявление закономерностей
        5. **Обучение модели**: Выбор и тренировка алгоритмов
        6. **Оценка модели**: Тестирование на независимых данных
        7. **Внедрение**: Интеграция модели в рабочую среду
        8. **Мониторинг**: Отслеживание производительности модели
        """)

    def _render_practical_part(self):
        st.header("4. Практическая часть")

        st.subheader("4.1 Выбор и описание набора данных")
        st.markdown("""
        Для практической реализации проекта был выбран открытый датасет «Titanic Survival Prediction Dataset».

        #### Характеристики датасета:
        - **Доступность и репутация**: Классический датасет в сообществе Data Science
        - **Соответствие задаче**: Содержит признаки и целевую переменную для бинарной классификации
        - **Интуитивная понятность**: Признаки не требуют узкоспециальных знаний
        - **Наличие вызовов**: Содержит пропущенные значения, требует предобработки

        #### Структура данных:
        - **PassengerId**: Порядковый номер записи
        - **Survived**: Целевая переменная (0 - не выжил, 1 - выжил)
        - **Pclass**: Класс билета (1, 2, 3)
        - **Name**: Имя пассажира
        - **Sex**: Пол
        - **Age**: Возраст
        - **SibSp**: Количество братьев/сестёр и супругов
        - **Parch**: Количество родителей и детей
        - **Ticket**: Номер билета
        - **Fare**: Стоимость билета
        - **Cabin**: Каюта
        - **Embarked**: Порт посадки
        """)

        st.subheader("4.2 Предобработка данных")
        st.markdown("""
        #### Обработка пропущенных значений:
        - **Age**: Заполнение медианным значением (устойчиво к выбросам)
        - **Embarked**: Заполнение наиболее частотным значением (модой)
        - **Cabin**: Исключение из анализа из-за большого количества пропусков

        #### Кодирование категориальных признаков:
        - **One-Hot Encoding**: Для признаков "Sex" и "Embarked"
        - Создание бинарных переменных для каждой категории
        - Исключение ложной порядковой зависимости

        #### Создание новых признаков:
        - Извлечение обращения (Title) из признака "Name"
        - Информация о возрасте, семейном положении и социальном статусе

        #### Масштабирование признаков:
        - **Стандартизация**: Приведение к среднему 0 и стандартному отклонению 1
        - Обеспечение стабильной работы градиентного спуска
        - Исключение доминирования признаков с большим диапазоном
        """)

        st.subheader("4.3 Разведочный анализ данных")
        st.markdown("""
        #### Анализ целевой переменной:
        - Изучение распределения классов (выжившие/погибшие)
        - Выявление дисбаланса классов

        #### Анализ взаимосвязей:
        - **Категориальные признаки**: Сгруппированные столбчатые диаграммы
        - **Количественные признаки**: Box-plot, гистограммы, графики плотности
        - **Матрица корреляций**: Тепловая карта линейных зависимостей

        #### Ключевые выводы EDA:
        - Пол является сильным предиктором выживаемости
        - Социально-экономический статус тесно связан с шансом на спасение
        - Семейные пассажиры имели отличные стратегии эвакуации
        - Подтверждение исторических фактов статистическими методами
        """)

        st.subheader("4.4 Обучение и оценка модели логистической регрессии")
        st.markdown("""
        #### Разделение данных:
        - **Обучающая выборка**: 75% данных
        - **Тестовая выборка**: 25% данных
        - **Стратификация**: Сохранение баланса целевой переменной

        #### Процесс обучения:
        - **Инициализация**: Нулевые значения весов и смещения
        - **Итеративное обновление**: Градиентный спуск на каждой эпохе
        - **Вычисление градиента**: Производная функции потерь по параметрам
        - **Оптимизация скорости обучения**: Баланс между сходимостью и стабильностью

        #### Оценка качества:
        - **Accuracy**: Доля правильных ответов
        - **Матрица ошибок**: Детальный анализ классификации
        - **Precision**: Точность положительных прогнозов
        - **Recall**: Полнота обнаружения положительных instances
        - **F1-мера**: Гармоническое среднее precision и recall

        #### Валидация робастности:
        - Варьирование начальных условий
        - Анализ стабильности метрик
        - Исследование систематических ошибок
        """)

        st.subheader("4.5 Создание интерфейса для взаимодействия с моделью")
        st.markdown("""
        #### Архитектура системы:
        - **Модульный принцип**: Разделение прогностического ядра и интерфейса
        - **Инкапсуляция логики**: Предобработка, прогноз, интерпретация
        - **Веб-интерфейс**: Кроссплатформенность и доступность

        #### Пользовательский интерфейс:
        - **Специализированные элементы**: Поля ввода, выпадающие списки, слайдеры
        - **Валидация данных**: Проверка полноты, типов и диапазонов
        - **Консистентность**: Те же преобразования, что и для тренировочных данных

        #### Процесс взаимодействия:
        1. Ввод исходных данных пользователем
        2. Комплексная валидация входных параметров
        3. Предобработка в соответствии с тренировочным pipeline
        4. Выполнение прогноза моделью
        5. Преобразование в бинарный прогноз через пороговое значение
        6. Предоставление развернутой интерпретации результата

        #### Обработка ошибок:
        - **Изящная деградация**: Корректная работа при нештатных ситуациях
        - **Многоуровневая система**: Обработка исключений на всех этапах
        - **Информативные сообщения**: Понятные уведомления об ошибках

        #### Документация и мониторинг:
        - **Комплексная документация**: Техническая и пользовательская
        - **Юзабилити-тестирование**: Оптимизация пользовательского пути
        - **Механизмы мониторига**: Отслеживание производительности в эксплуатации
        """)

    def _render_results_conclusions(self):
        st.header("5. Результаты и выводы")

        st.subheader("Достигнутые результаты")
        st.markdown("""
        #### Практическая реализация:
        - Создана полнофункциональная система бинарной классификации
        - Реализован тщательный процесс предобработки данных
        - Доказана эффективность градиентного спуска для обучения моделей

        #### Исследовательская ценность:
        - Выявлены статистические закономерности в исторических данных
        - Подтверждена обоснованность методов ML для решения подобных задач
        - Создан интерактивный интерфейс для демонстрации возможностей ML

        #### Научное влияние:
        - Изучение работ российского учёного Дмитрия Ветрова
        - Применение вероятностных моделей и методов объяснимого ИИ
        - Подтверждение важности фундаментальных математических знаний
        """)

        st.subheader("Перспективы развития")
        st.markdown("""
        #### Направления профессионального роста:
        1. **Углубление математической подготовки**:
           - Линейная алгебра
           - Теория вероятностей
           - Математическая статистика

        2. **Освоение сложных архитектур**:
           - Глубокое обучение
           - Нейронные сети
           - Компьютерное зрение
           - Обработка естественного языка

        3. **Участие в профессиональном сообществе**:
           - Платформы: Kaggle, Habr
           - Публикация исследовательских статей
           - Обмен знаниями с российским IT-сообществом

        #### Карьерные амбиции:
        - Становление full-stack специалистом в области данных
        - Сочетание экспертизы в математике, программировании и предметной области
        - Понимание бизнес-контекста и эффективная коммуникация результатов
        """)

        st.subheader("Заключение")
        st.markdown("""
        Данный проект стал важным этапом в освоении машинного обучения, продемонстрировав возможность
        самостоятельного достижения значимых результатов через систематическое изучение теории и её
        практическую апробацию.

        **Приобретённые знания и навыки** создают прочную основу для дальнейшего профессионального
        развития в одной из наиболее перспективных и быстроразвивающихся областей современной информатики.

        Все материалы проекта доступны в открытом репозитории:
        [GitHub](https://github.com/alexeev-prog/ML-DS_learning_article)
        """)

    def _render_sources(self):
        st.header("Источники")

        st.subheader("Литература")
        sources = [
            {
                "title": "Грокаем машинное обучение",
                "author": "Серрано, Л.",
                "publisher": "Питер, 2022",
                "pages": "480 с.",
            },
            {
                "title": "Грокаем алгоритмы. Иллюстрированное пособие для программистов и любопытствующих",
                "author": "Бхаргава, А.",
                "publisher": "Питер, 2024",
                "pages": "290 с.",
            },
            {
                "title": "Грокаем глубокое обучение",
                "author": "Траск, Э.",
                "publisher": "Питер, 2023",
                "pages": "304 с.",
            },
            {
                "title": "Грокаем алгоритмы",
                "author": "Бхаргава, А.",
                "publisher": "Питер, 2025",
                "pages": "304 с.",
            },
        ]

        for i, source in enumerate(sources, 1):
            st.write(
                f"{i}. {source['author']} {source['title']}. --- {source['publisher']}. --- {source['pages']}"
            )

        st.subheader("Электронные ресурсы")
        electronic_sources = [
            {
                "name": "Titanic Dataset - Machine Learning from Disaster",
                "platform": "Kaggle, 2024",
                "url": "https://www.kaggle.com/datasets/yasserh/titanic-dataset",
            },
            {
                "name": "Профиль на Habr",
                "platform": "Habr",
                "url": "https://habr.com/ru/users/DrArgentum",
            },
            {
                "name": "Репозиторий проекта",
                "platform": "GitHub",
                "url": "https://github.com/alexeev-prog/ML-DS_learning_article",
            },
        ]

        for source in electronic_sources:
            st.write(f"- **{source['name']}** [{source['platform']}]({source['url']})")

    def _render_technical_documentation(self):
        st.header("Техническая документация")

        st.subheader("Архитектура системы")
        st.markdown("""
        #### Модульная структура:
        - **core.services**: Бизнес-логика и сервисы
        - **data.analysis**: Визуализация и анализ данных
        - **data.preprocessing**: Подготовка и очистка данных
        - **data.repositories**: Работа с источниками данных
        - **entities.core**: Основные сущности и модели данных

        #### Используемые технологии:
        - **Python 3.8+**: Основной язык программирования
        - **Streamlit**: Фреймворк для веб-интерфейса
        - **Pandas/NumPy**: Обработка и анализ данных
        - **Plotly**: Интерактивная визуализация
        - **Scikit-learn**: Машинное обучение (внутренние реализации)
        """)

        st.subheader("Основные компоненты")
        st.markdown("""
        #### DataTableComponent:
        - Отображение табличных данных с поиском и сортировкой
        - Пагинация и экспорт в CSV

        #### PlotComponent:
        - Визуализация графиков и диаграмм
        - Обновление и скачивание изображений

        #### ModelMetricsComponent:
        - Отображение метрик производительности модели
        - Матрица неточностей и детальная статистика

        #### PredictionFormComponent:
        - Форма ввода данных для предсказания
        - Валидация пользовательского ввода
        """)

        st.subheader("Страницы приложения")
        st.markdown("""
        - **Главная**: Обзор проекта и технического стека
        - **Анализ данных**: Загрузка и исследование датасета
        - **Обучение модели**: Конфигурация и тренировка модели
        - **Объяснение модели**: Анализ и интерпретация результатов
        - **Предсказания**: Интерактивное использование модели
        - **Документация**: Теоретическая и техническая документация
        """)


class HomePage:
    def render(self, state) -> None:
        st.title("🚢 Модель данных выживания на титанике")
        st.markdown("---")

        st.header("Проект по анализу данных и машинному обучению")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
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
            """)

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
            st.markdown("""
            - Кастомная логистическая регрессия
            - Градиентный спуск
            - NumPy/Pandas
            """)

        with tech_cols[1]:
            st.subheader("📊 Визуализация")
            st.markdown("""
            - Matplotlib/Seaborn
            - Plotly
            - Streamlit
            """)

        with tech_cols[2]:
            st.subheader("🏗️ Архитектура")
            st.markdown("""
            - Чистая архитектура
            - SOLID
            - Паттерны ООП
            - ACID
            - DRY, KISS
            """)

        with tech_cols[3]:
            st.subheader("📈 Аналитика")
            st.markdown("""
            - Визуализация EDA
            - Метрики модели
            - Статистический анализ
            - Анализ решений модели
            """)

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

    def _render_training_controls(self) -> dict[str, Any]:
        col1, col2, col3 = st.columns(3)

        with col1:
            learning_rate = st.slider("Скорость обучения", 0.001, 0.5, 0.05, 0.001)
            epochs = st.number_input("Эпохи", 1000, 100000, 3000, 100)
            polynomial_degree = st.selectbox("Степень полинома", [1, 2, 3], index=1)

        with col2:
            test_size = st.slider("Размер теста", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Случайное состояние", 0, 100, 42)
            convergence_tol = st.number_input(
                "Допуск сходимости",
                1e-8,
                1e-4,
                1e-4,
                1e-8,
                format="%.8f",
            )

        with col3:
            lambda_reg = st.slider("Регуляризация", 0.0, 1.0, 0.01, 0.01)
            use_adam = st.checkbox("Использовать Adam оптимизатор", value=True)
            beta1 = st.slider("Beta1 (Adam)", 0.8, 0.999, 0.9, 0.001)
            beta2 = st.slider("Beta2 (Adam)", 0.8, 0.999, 0.999, 0.001)

        return {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "test_size": test_size,
            "random_state": random_state,
            "convergence_tol": convergence_tol,
            "lambda_reg": lambda_reg,
            "polynomial_degree": polynomial_degree,
            "use_adam": use_adam,
            "beta1": beta1,
            "beta2": beta2,
        }

    def _train_model(self, state: AppState, config: dict[str, Any]) -> None:
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
                    lambda_reg=config["lambda_reg"],
                    polynomial_degree=config["polynomial_degree"],
                    use_adam=config["use_adam"],
                    beta1=config["beta1"],
                    beta2=config["beta2"],
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
                st.metric("Степень полинома", training_result.config.polynomial_degree)
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
                line={"color": "blue", "width": 2},
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
            st.error(f"Детальная ошибка: {e!s}")

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
    def __init__(self, app_config: dict | None = None):
        self.pages = {
            "Главная": HomePage(),
            "Анализ данных": DataAnalysisPage(),
            "Обучение модели": ModelTrainingPage(),
            "Объяснение модели": ModelExplanationPage(),
            "Предсказания": PredictionPage(),
            "Документация": DocumentationPage(),
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


if __name__ == "__main__":
    app = TitanicApp()
    app.run()
