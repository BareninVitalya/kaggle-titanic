import joblib

from .config import TARGET_COL, MODELS_DIR, USE_OPENFE, OPENFE_PARAMS
from .data import load_train, save_processed
from .features import TitanicFeatures
from .modeling import build_model
from .evaluate import cv_scores
from .logging_utils import log_experiment
from .openfe_stage import OpenFEStage

from typing import Any
import pandas as pd
import numpy as np


def run_experiment(
    model_name: str = "logreg",
    params: dict | None = None,
    run_id: str | None = None,
    save_model: bool = True,
    use_openfe: bool = USE_OPENFE,
    openfe_params: dict | None = None,
) -> tuple[float, float]:
    """Полный цикл: загрузка → фичи → OpenFE → модель → CV → лог → fit → сохранение."""
    df = load_train()
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    fe = TitanicFeatures()
    X = fe.transform(X_raw)

    openfe_stage = None
    if use_openfe:
        resolved = {**OPENFE_PARAMS, **(openfe_params or {})}
        openfe_stage = OpenFEStage(model_name=model_name, **resolved)
        X = openfe_stage.fit_transform(X, y)

    model = build_model(model_name, X, params=params)
    mean, std, scores = cv_scores(model, X, y)
    log_experiment(model_name, mean, std, params or {})

    print(f"CV {model_name}: mean={mean:.4f} std={std:.4f}")

    model.fit(X, y)

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = model_name

        model_path = MODELS_DIR / f"{model_name}_{run_id}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        if openfe_stage is not None:
            stage_path = MODELS_DIR / f"openfe_stage_{run_id}.pkl"
            openfe_stage.save(stage_path)

    df_proc = X.copy()
    df_proc[TARGET_COL] = y
    save_processed(df_proc, name="train_clean.csv")

    return mean, std

def quick_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logreg",
    params: dict[str, Any] | None = None,
    transform_off: bool = True
) -> None:

    model = build_model(model_name, X, params, transform_off)
    mean, std, scores = cv_scores(model, X, y)
    print(f"CV {model_name}: mean={mean} std={std}")
    return None


if __name__ == "__main__":
    run_experiment()