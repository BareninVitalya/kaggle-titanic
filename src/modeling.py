from typing import Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .config import SEED,DEFAULT_LOGREG_PARAMS, TARGET_COL
from .features import TitanicFeaturesTransformer
from .data import load_train, load_test


def build_matual_info_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Строим общий препроцессор для числовых и категориальных признаков."""

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Строим общий препроцессор для числовых и категориальных признаков."""

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor


def build_logreg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        X_fe = X.copy()
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="linear"
        )
        X_fe = fe.fit_transform(X)
    
    pre = build_preprocessor(X_fe)

    base_params = DEFAULT_LOGREG_PARAMS.copy()
    base_params.update(params or {})


    clf = LogisticRegression(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])


def build_rf_model(X: pd.DataFrame, params: Optional[Dict] = None) -> Pipeline:
    pre = build_preprocessor(X)
    base_params = dict(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        random_state=SEED,
    )
    if params:
        base_params.update(params)

    clf = RandomForestClassifier(**base_params)
    return Pipeline([("prep", pre), ("model", clf)])


def build_gb_model(X: pd.DataFrame, params: Optional[Dict] = None) -> Pipeline:
    pre = build_preprocessor(X)
    base_params = dict(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        random_state=SEED,
    )
    if params:
        base_params.update(params)

    clf = GradientBoostingClassifier(**base_params)
    return Pipeline([("prep", pre), ("model", clf)])


def build_model(name: str, X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:
    """Фабрика моделей по имени."""
    name = name.lower()
    if name in ("logreg", "lr", "logistic"):
        return build_logreg_model(X, params, transform_off)
    if name in ("rf", "random_forest"):
        return build_rf_model(X, params)
    if name in ("gb", "gboost", "gradboost"):
        return build_gb_model(X, params)
    raise ValueError(f"Unknown model name: {name}")

def train_model(model_name: str = 'logreg'):
    df_train = load_train()

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]

    model = build_model(model_name, X_train)
    model.fit(X_train, y_train)

    return model