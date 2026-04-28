from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import SEED, N_SPLITS


def cv_scores(
    model,
    X,
    y,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: str = "accuracy",
) -> Tuple[float, float, np.ndarray]:
    """Возвращает mean, std, и массив score'ов по StratifiedKFold, как в ноутбуке.[file:1]"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return float(scores.mean()), float(scores.std()), scores


def compare_models(
    builders: Dict[str, Callable[[pd.DataFrame], object]],
    X,
    y,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Обгоняет список моделей и возвращает таблицу с их mean/std score."""
    rows = []
    for name, builder in builders.items():
        model = builder(X)
        mean, std, _ = cv_scores(model, X, y, n_splits=n_splits, seed=seed, scoring=scoring)
        rows.append({"model": name, "mean": mean, "std": std})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df