from typing import Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


from .config import (
    SEED,
    DEFAULT_LOGREG_PARAMS,
    TARGET_COL,
    NUM_FEATURES,
    CAT_FEATURES,
    DEFAULT_TREE_PARAMS,
    DEFAULT_KNN_PARAMS,
    DEFAULT_RF_PARAMS,
    DEFAULT_CATBOOST_PARAMS,
    DEFAULT_LGBM_PARAMS,
    DEFAULT_XGB_PARAMS,
)

from .features import TitanicFeaturesTransformer
from .data import load_train, load_test, save_processed


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


def build_preprocessor(X: Optional[pd.DataFrame] = None, transform_off: bool = False) -> ColumnTransformer:
    """Строим общий препроцессор для числовых и категориальных признаков."""
    if not transform_off and X is not None:
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        num_cols = NUM_FEATURES
        cat_cols = CAT_FEATURES

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

def build_catboost_preprocessor(X: Optional[pd.DataFrame] = None, transform_off: bool = False) -> ColumnTransformer:
    """CatBoost препроцессор"""
    if not transform_off and X is not None:
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        num_cols = NUM_FEATURES
        cat_cols = CAT_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor


def build_logreg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="linear"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_LOGREG_PARAMS.copy()

    if params:
        base_params.update(params)


    clf = LogisticRegression(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_knn_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="knn"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_KNN_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = KNeighborsClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_tree_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="tree"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_TREE_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = DecisionTreeClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_rf_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="rf"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_RF_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = RandomForestClassifier(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_catboost_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="rf"
        )
        pre = build_preprocessor()
        cat_cols = CAT_FEATURES

    base_params = DEFAULT_CATBOOST_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = CatBoostClassifier(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_lgbm_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="lgbm"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_LGBM_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = LGBMClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_xgb_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:

    if transform_off:
        fe = FunctionTransformer(lambda x: x)
        pre = build_preprocessor(X)
    else:
        fe = TitanicFeaturesTransformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            model_type="xgb"
        )
        pre = build_preprocessor()

    base_params = DEFAULT_XGB_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = XGBClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])



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
    if name in ("knn", "kneighbors", "k_neighbors"):
        return build_knn_model(X, params, transform_off)
    if name in ("tree", "dt", "decision_tree"):
        return build_tree_model(X, params, transform_off)
    if name in ("rf", "random_forest"):
        return build_rf_model(X, params, transform_off)
    if name in ("catboost", "cat"):
        return build_catboost_model(X, params, transform_off)
    if name in ("lgbm", "lightgbm"):
        return build_lgbm_model(X, params, transform_off)
    if name in ("xgb", "xgboost"):
        return build_xgb_model(X, params, transform_off)
    raise ValueError(f"Unknown model name: {name}")

def train_model(model_name: str = 'logreg', params: Optional[Dict] = None):
    df_train = load_train()

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]

    model = build_model(model_name, X_train)
    model.fit(X_train, y_train)

    return model

def predict_and_save_titanic(model, test_data: pd.DataFrame, file_name: str = "submission"):

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": model.predict(test_data).astype(int),
    })

    save_processed(submission, f"{file_name}.csv")
