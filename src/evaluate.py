from typing import Dict, Callable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from .config import SEED, N_SPLITS


def cv_scores(
    model,
    X,
    y,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: str = "accuracy",
    return_train_score: bool = False
) -> Tuple[float, float, np.ndarray, Union[float, np.ndarray]]:
    """Возвращает mean, std, и массив score'ов по StratifiedKFold, как в ноутбуке.[file:1]"""
    if scoring in ['neg_median_absolute_error']:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    val_scores = result["test_score"]
    val_mean   = float(val_scores.mean())
    val_std    = float(val_scores.std())
    train_data = float(result["train_score"].mean()) if not return_train_score else result["train_score"]

    return val_mean, val_std, val_scores, train_data


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
        mean, std, _, _ = cv_scores(model, X, y, n_splits=n_splits, seed=seed, scoring=scoring)
        rows.append({"model": name, "mean": mean, "std": std})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df