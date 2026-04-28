import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from openfe import OpenFE, transform

from .config import MODELS_DIR, SEED
from .evaluate import cv_scores
from .modeling import build_model


def run_openfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 50,      # сколько кандидатов генерировать
    n_jobs: int = 1,
    seed: int = SEED,
) -> list:
    """
    Запускает OpenFE на тренировочных данных.
    Возвращает список feature-объектов, отсортированных по важности.
    """
    ofe = OpenFE()
    features = ofe.fit(
        data=X_train,
        label=y_train,
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,       # печатает прогресс
        feature_boosting=False,  # True — дольше но точнее
    )
    return features[:n_features]


def apply_openfe(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Применяет отобранные OpenFE-признаки к train и test."""
    X_train_new, X_test_new = transform(X_train, X_test, features, n_jobs=1)
    return X_train_new, X_test_new


def save_features(features: list, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} features to {path}")


def load_features(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_importance_report(features, top_n=20):
    rows = []
    for i, feat in enumerate(features[:top_n]):
        # Правильный способ получить описание признака
        try:
            feature_str = feat.root.get_fnode()  # полное дерево операций
        except Exception:
            try:
                feature_str = feat.root.__class__.__name__  # хотя бы тип
            except Exception:
                feature_str = "unknown"

        rows.append({
            "rank":    i + 1,
            "operator": feat.name,    # тип операции: /, *, log, GroupByThenRank
            "feature":  feature_str,
        })
    return pd.DataFrame(rows)


def ablation_openfe(
    X_base: pd.DataFrame,
    X_with_ofe: pd.DataFrame,
    y: pd.Series,
    features: list,
    model_name: str = "logreg",
    step: int = 5,
) -> pd.DataFrame:
    """
    Поэтапно добавляет признаки (шаг step) и смотрит как меняется CV score.
    Позволяет найти оптимальное кол-во OFE-признаков.
    """
    base_score, base_std, _ = cv_scores(build_model(model_name, X_base), X_base, y)

    rows = [{"n_features": 0, "mean": base_score, "std": base_std, "delta": 0.0}]

    ofe_cols = [c for c in X_with_ofe.columns if c not in X_base.columns]

    for n in range(step, len(ofe_cols) + 1, step):
        cols_to_add = ofe_cols[:n]
        X_exp = pd.concat([X_base, X_with_ofe[cols_to_add]], axis=1)
        mean, std, _ = cv_scores(build_model(model_name, X_exp), X_exp, y)
        rows.append({
            "n_features": n,
            "mean": mean,
            "std":  std,
            "delta": round(mean - base_score, 5),
        })
        print(f"  n={n:3d}  mean={mean:.4f}  delta={mean - base_score:+.5f}")

    return pd.DataFrame(rows)