# Kaggle Titanic

## Обзор

Проект решает задачу бинарной классификации Kaggle Titanic: предсказание выживаемости пассажиров. Кодовая база организована как Python-пакет `src/` с полным разделением ответственности между модулями. Ноутбук `summary.ipynb` является демонстрационной точкой входа — в нём показываются результаты экспериментов, но вся логика сосредоточена в `src/`.

**Метрика:** accuracy (StratifiedKFold, N_SPLITS=5, SEED=42).

***

## Структура проекта

```
kaggle-titanic/
├── src/
│   ├── config.py          # Пути, константы, гиперпараметры моделей
│   ├── data.py            # Загрузка и сохранение данных
│   ├── features.py        # Feature engineering (трансформеры)
│   ├── modeling.py        # Фабрика пайплайнов и моделей
│   ├── evaluate.py        # CV-оценка и сравнение моделей
│   ├── ensemble.py        # Ансамблирование (SoftEnsemble, Voting, Stacking)
│   ├── feature_search.py  # OpenFE + ablation по авто-фичам
│   ├── logging_utils.py   # Сохранение и чтение логов экспериментов
│   └── nn_model.py        # PyTorch DNN-классификатор
│
├── data/
│   ├── raw/               # train.csv, test.csv (исходные данные Kaggle)
│   └── processed/         # сохранённые submission-файлы
│
├── models/                # сохранённые joblib-модели (age_model.joblib и др.)
├── logs/                  # experiments.log (JSON-строки с результатами CV)
├── output/                # итоговые submission.csv
│
└── summary.ipynb          # Демонстрационный ноутбук с экспериментами
```

***

## Описание модулей

### `config.py` — центральный конфиг проекта

Единственное место, где задаются все пути, константы и дефолтные параметры моделей.

**Ключевые переменные:**

| Переменная | Описание |
|---|---|
| `SEED = 42` | Глобальный random seed для воспроизводимости |
| `N_SPLITS = 5` | Количество фолдов в StratifiedKFold |
| `TARGET_COL = "Survived"` | Целевая переменная |
| `NOISE_FEATURES` | Список признаков, удалённых по результатам ablation-анализа |
| `NUM_FEATURES` / `CAT_FEATURES` | Финальный набор числовых и категориальных признаков |
| `DEFAULT_*_PARAMS` | Дефолтные гиперпараметры для каждой модели (LogReg, KNN, Tree, RF, CatBoost, LGBM, XGB, DNN) |
| `RANDOM_SEARCH_SPACE` | Пространство поиска для RandomizedSearchCV |
| `USE_OPENFE`, `OPENFE_PARAMS` | Флаг и параметры для автоматического feature engineering |
| `PCLASS_SEX_RANK` | Ранговая кодировка взаимодействия Pclass×Sex (результат EDA) |

***

### `data.py` — загрузка и сохранение данных

Минимальный модуль с тремя функциями.

| Функция | Что делает |
|---|---|
| `load_train()` | Читает `data/raw/train.csv` → `pd.DataFrame` |
| `load_test()` | Читает `data/raw/test.csv` → `pd.DataFrame` |
| `save_processed(df, name)` | Сохраняет DataFrame в `data/processed/` |

***

### `features.py` — feature engineering

Содержит иерархию классов-трансформеров. Все ручные признаки создаются здесь.

**Иерархия классов:**

```
TitanicFeaturesBase          # Базовый класс: общие методы извлечения признаков
├── AgeFeatures              # Трансформер для задачи регрессии Age
│   └── AgeTransformer       # sklearn-совместимая обёртка (Pipeline-ready)
└── SurvivalFeatures         # Трансформер для задачи классификации Survived
    └── SurvivalTransformer  # sklearn-совместимая обёртка (Pipeline-ready)
```

**Что создаётся в базовом классе (`TitanicFeaturesBase`):**

| Признак | Метод | Описание |
|---|---|---|
| `Title` | `_extract_title` | Извлечение социального титула из имени; редкие → `Rare` |
| `familysize` | `_add_family_size` | `SibSp + Parch + 1` |
| `isalone` | `_add_family_size` | 1 если пассажир путешествует один |
| `ticketgroupsize` | `_add_ticket_group_size` | Размер группы по одному номеру билета |
| `TicketPrefix` | `_add_ticket_prefix` | Буквенный префикс номера билета |
| `CabinDeck` | `_add_cabin_deck` | Первая буква каюты (палуба) |
| `Sex` (encoded) | `_encode_sex` | `male→0`, `female→1` |

**Дополнительные признаки в `SurvivalFeatures`:**

| Признак | Метод | Описание |
|---|---|---|
| `Pclass_Sex` | `_add_pclass_sex_feature` | `Pclass × Sex` — взаимодействие класса и пола |
| `Age_bin` | `_bin_age_feature` | Бинаризация возраста в 8 бакетов `[0,10,20...80]` |
| `Fare_bin` | `_bin_fare_log_feature` / `_bin_fare_feature` | Бинаризация стоимости билета |

**Логика заполнения пропусков Age:**
- Основной путь: медиана по группе `Title` (`_impute_age_median_of_title`)
- Альтернативный: отдельная регрессионная модель `age_model.joblib`

***

### `modeling.py` — фабрика моделей и пайплайнов

Центральный модуль сборки. Все модели строятся как sklearn `Pipeline` из трёх шагов: `feat → prep → model`.

**Архитектура пайплайна:**
```
Pipeline:
  feat  → SurvivalTransformer / AgeTransformer / FunctionTransformer(identity)
  prep  → ColumnTransformer (StandardScaler для числовых, OHE/Ordinal для категориальных)
  model → классификатор / регрессор
```

**Ключевые функции:**

| Функция | Описание |
|---|---|
| `get_transformer(...)` | Выбирает трансформер признаков (SurvivalTransformer или AgeTransformer) |
| `build_preprocessor(X)` | Собирает ColumnTransformer (числовые: impute+scale, категориальные: impute+OHE) |
| `build_catboost_preprocessor(X)` | Препроцессор для CatBoost (без OHE — только impute) |
| `build_mutual_info_preprocessor(X)` | Препроцессор для расчёта mutual information (OrdinalEncoder) |
| `build_<name>_model(X, params)` | Фабричные функции для каждой модели |
| `build_model(name, X, params)` | Единая фабрика по строковому имени модели |
| `train_model(model_name, params)` | Обучение модели на train.csv и возврат пайплайна |
| `predict_and_save_titanic(model, test_data)` | Генерация и сохранение submission.csv |

**Поддерживаемые имена моделей:**

| Ключи | Модель |
|---|---|
| `"logreg"`, `"lr"`, `"logistic"` | LogisticRegression |
| `"knn"`, `"kneighbors"` | KNeighborsClassifier |
| `"tree"`, `"dt"` | DecisionTreeClassifier |
| `"rf"`, `"random_forest"` | RandomForestClassifier |
| `"catboost"`, `"cat"` | CatBoostClassifier |
| `"lgbm"`, `"lightgbm"` | LGBMClassifier |
| `"xgb"`, `"xgboost"` | XGBClassifier |
| `"dnn"`, `"mlp"`, `"nn"` | TitanicDNNClassifier (PyTorch) |
| `"rf_reg"` | RandomForestRegressor (для задачи Age) |

> Параметр `transform_off=True` позволяет передать уже трансформированные данные напрямую в препроцессор, минуя `SurvivalTransformer`. Это используется в ансамблях и экспериментах с OpenFE.

***

### `evaluate.py` — кросс-валидация и сравнение моделей

Две функции для воспроизводимой оценки качества.

| Функция | Описание |
|---|---|
| `cv_scores(model, X, y, ...)` | Возвращает `(mean, std, scores_array, train_mean)` по StratifiedKFold |
| `compare_models(builders, X, y)` | Принимает словарь `{name: builder_fn}`, возвращает DataFrame с mean/std отсортированный по качеству |

`cv_scores` автоматически переключается на `KFold` для регрессионных метрик (`neg_median_absolute_error`).

***

### `ensemble.py` — ансамблирование моделей

Три класса ансамблей, каждый совместим с `build_model` из `modeling.py`.

#### `SoftEnsemble`
Кастомный soft-voting с поддержкой весов и настраиваемого порога бинаризации.
- `.build(X)` — собирает базовые модели
- `.fit(X, y)` — обучает все модели
- `.predict_proba(X)` — взвешенное среднее вероятностей
- `.predict(X, threshold)` — бинаризация по порогу (default 0.5)
- `.predict_test(df_test, file_name)` — генерация submission

#### `VotingEnsemble`
Обёртка над sklearn `VotingClassifier`. Используется когда нужен полный sklearn-совместимый интерфейс (например, для GridSearch поверх ансамбля).

#### `StackingEnsemble`
OOF-stacking с мета-моделью. Работает в два этапа:
1. `.build_oof_data(X, y, X_test)` — строит OOF-признаки через StratifiedKFold
2. `.fit(X, y, X_test)` — обучает мета-модель (`ridge` / `logreg`) на OOF-признаках

***

### `feature_search.py` — автоматический feature engineering (OpenFE)

Модуль для запуска и анализа автоматически генерируемых признаков через библиотеку [OpenFE](https://github.com/IIIS-ML/OpenFE).

| Функция | Описание |
|---|---|
| `run_openfe(X_train, y_train, n_features)` | Запускает OpenFE и возвращает топ-N признаков |
| `apply_openfe(X_train, X_test, features)` | Применяет признаки к train и test |
| `save_features(features, path)` | Сохраняет признаки через pickle |
| `load_features(path)` | Загружает сохранённые признаки |
| `feature_importance_report(features, top_n)` | Таблица с описанием топ-N признаков |
| `ablation_openfe(X_base, X_with_ofe, y, features, step)` | Поэтапное добавление признаков с измерением прироста CV score |

***

### `logging_utils.py` — логирование экспериментов

Простая система записи результатов CV в JSON-файл без зависимостей от MLflow или аналогов.

| Функция | Описание |
|---|---|
| `log_experiment(name, score, std, params, col_names)` | Записывает результат эксперимента в `logs/experiments.log` (одна JSON-строка на запись) |
| `load_experiments(logfile)` | Загружает лог как DataFrame с колонками `timestamp, name, score, std, params, col_names` |

***

### `nn_model.py` — PyTorch DNN-классификатор

Реализация `TitanicDNNClassifier` — sklearn-совместимого классификатора на основе полносвязной нейросети (MLP). Принимает числовой вход после `build_preprocessor(sparse_output=False)`. Архитектура и гиперпараметры задаются через `DEFAULT_DNN_PARAMS` в `config.py`.

***

## Как запустить проект

```bash
# 1. Клонировать репозиторий
git clone https://github.com/BareninVitalya/kaggle-titanic.git
cd kaggle-titanic

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Положить данные Kaggle в data/raw/
#    train.csv и test.csv скачиваются с kaggle.com/competitions/titanic

# 4. Открыть ноутбук
jupyter notebook summary.ipynb
```

> Для воспроизводимости результатов достаточно запустить все ячейки сверху вниз. Все параметры зафиксированы в `config.py (SEED=42, N_SPLITS=5)`.