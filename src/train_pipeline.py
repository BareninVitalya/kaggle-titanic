from pathlib import Path

import joblib
import pandas as pd

from .config import TARGET_COL, PROCESSED_DATA_DIR, MODELS_DIR
from .data import load_train, save_processed
from .features import TitanicFeatures
from .modeling import build_model
from .evaluate import cv_scores
from .logging_utils import log_experiment


def run_experiment(
    model_name: str = "logreg",
    params: dict | None = None,
    run_id: str | None = None,
    save_model: bool = True,
) -> tuple[float, float]:
    """Полный цикл: загрузка → фичи → модель → CV → лог → fit → сохранение."""
    # 1. Загрузка
    df = load_train()
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 2. Фичи
    fe = TitanicFeatures()
    X = fe.transform(X_raw)

    # 3. Модель
    model = build_model(model_name, X, params=params)

    # 4. CV
    mean, std, scores = cv_scores(model, X, y)
    log_experiment(model_name, mean, std, params or {})

    print(f"CV {model_name}: mean={mean:.4f} std={std:.4f}")

    # 5. Финальное обучение на всех данных
    model.fit(X, y)

    # 6. Сохранение модели
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = model_name
        model_path = MODELS_DIR / f"{model_name}_{run_id}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    # 7. Сохранить очищенный train для EDA/heatmap (как в ноутбуке) [file:1]
    df_proc = X.copy()
    df_proc[TARGET_COL] = y
    save_processed(df_proc, name="train_clean.csv")

    return mean, std


if __name__ == "__main__":
    run_experiment()