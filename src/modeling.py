from typing import Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from .nn_model import TitanicDNNClassifier


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
    DEFAULT_DNN_PARAMS,
    DEFAULT_RF_REG_PARAMS,
    MODELS_DIR
)

from .features import AgeTransformer, SurvivalTransformer, TitanicTransformerMixin
from .data import load_train, load_test, save_processed

def get_transformer(
        drop_sibsp_parch: bool = True,
        use_log_fare: bool = True,
        use_age_bins: bool = True,
        use_fare_bins: bool = True,
        use_pclass_sex: bool = True,
        age_model_path: str = rf"{MODELS_DIR}\age_model.joblib",
        age_prediction_model: bool = False,
        is_fit: bool = False,
        transform_off: bool = False
) -> TransformerMixin:
    if transform_off:
        return FunctionTransformer(lambda x: x)
    else:
        if age_prediction_model:
            return AgeTransformer(
                use_log_fare=use_log_fare,
                is_fit=is_fit,
            )
        else:
            return SurvivalTransformer(
                drop_sibsp_parch=drop_sibsp_parch,
                use_log_fare=use_log_fare,
                use_age_bins=use_age_bins,
                use_fare_bins=use_fare_bins,
                use_pclass_sex=use_pclass_sex,
                age_model_path=age_model_path,
            )


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

def build_preprocessor(X: Optional[pd.DataFrame] = None, transform_off: bool = False, sparse_output: bool = True) -> ColumnTransformer:
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
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
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_LOGREG_PARAMS.copy()

    if params:
        base_params.update(params)


    clf = LogisticRegression(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_knn_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_KNN_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = KNeighborsClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_tree_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_TREE_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = DecisionTreeClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_rf_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_RF_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = RandomForestClassifier(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_catboost_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()
        cat_cols = CAT_FEATURES

    base_params = DEFAULT_CATBOOST_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = CatBoostClassifier(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_rf_reg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=False,
            use_fare_bins=True,
            use_pclass_sex=False,
            age_prediction_model=True,
            is_fit=False, # При обучении модели на Age -> True
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_RF_REG_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = RandomForestRegressor(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_lgbm_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
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
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_XGB_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = XGBClassifier(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])



def build_dnn_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:
    """
    Создаёт TitanicDNNClassifier на основе DEFAULT_DNN_PARAMS + overrides из params.

    DNN ожидает числовой вход — используй transform_off=True в quick_experiment,
    чтобы передавать уже подготовленный X напрямую.

    Пример:
        model = build_model("dnn", X_proc, params={"hidden_dims": [128, 64], "dropout": 0.2})
        quick_experiment(X_proc, y, model_name="dnn", transform_off=True)
    """
    cfg = DEFAULT_DNN_PARAMS.copy()
    if params:
        cfg.update(params)

    fe = get_transformer(
        use_log_fare=True,
        use_age_bins=True,
        use_fare_bins=True,
        use_pclass_sex=True,
        transform_off=False
    )
    pre = build_preprocessor(sparse_output=False)

    clf = TitanicDNNClassifier(
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
        optimizer=cfg["optimizer"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        scheduler=cfg["scheduler"],
        scheduler_params=cfg["scheduler_params"],
        loss_fn=cfg["loss_fn"],
        random_state=SEED,
    )

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])


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
    if name in ("dnn", "mlp", "nn"):
        return build_dnn_model(X, params, transform_off)
    if name in ("rf_reg"):
        return build_rf_reg_model(X, params, transform_off)
    raise ValueError(f"Unknown model name: {name}")

def train_model(model_name: str = 'logreg', params: Optional[Dict] = None, train_data: pd.DataFrame = None, transform_off: bool = False):
    if transform_off and train_data is not None:
        df_train = train_data
    else:
        df_train = load_train()
        transform_off = False

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]

    model = build_model(model_name, X_train, params, transform_off=transform_off)
    model.fit(X_train, y_train)

    return model

def predict_and_save_titanic(model, test_data: pd.DataFrame, file_name: str = "submission"):

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": model.predict(test_data).astype(int),
    })

    save_processed(submission, f"{file_name}.csv")

def predict_and_save_titanic2(model, ids_data: pd.DataFrame, test_data: pd.DataFrame, file_name: str = "submission"):

    submission = pd.DataFrame({
        "PassengerId": ids_data["PassengerId"],
        "Survived": model.predict(test_data).astype(int),
    })

    save_processed(submission, f"{file_name}.csv")