from typing import Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .config import SEED,DEFAULT_LOGREG_PARAMS


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


def build_logreg_model(X: pd.DataFrame, params: Optional[Dict] = None) -> Pipeline:
    pre = build_preprocessor(X)
    base_params = DEFAULT_LOGREG_PARAMS
    if params:
        base_params.update(params)

    clf = LogisticRegression(**base_params)
    return Pipeline([("prep", pre), ("model", clf)])


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


def build_model(name: str, X: pd.DataFrame, params: Optional[Dict] = None) -> Pipeline:
    """Фабрика моделей по имени."""
    name = name.lower()
    if name in ("logreg", "lr", "logistic"):
        return build_logreg_model(X, params)
    if name in ("rf", "random_forest"):
        return build_rf_model(X, params)
    if name in ("gb", "gboost", "gradboost"):
        return build_gb_model(X, params)
    raise ValueError(f"Unknown model name: {name}")