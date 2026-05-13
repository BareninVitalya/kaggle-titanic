import joblib

from .config import TARGET_COL, MODELS_DIR, USE_OPENFE, OPENFE_PARAMS, NUM_FEATURES, CAT_FEATURES, SEED
from .data import load_train, save_processed
from .modeling import build_model
from .evaluate import cv_scores
from .logging_utils import log_experiment
from .openfe_stage import OpenFEStage
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


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
    mean, std, scores, _ = cv_scores(model, X, y)
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
    transform_off: bool = True,
    scoring: str = 'accuracy',
    print_log: bool = False,
) -> None:

    model = build_model(model_name, X, params, transform_off)
    mean, std, scores, train_mean = cv_scores(model, X, y, scoring=scoring, return_train_score=True)
    if transform_off:
        cols = X.columns.values.tolist()
    else:
        cols = NUM_FEATURES + CAT_FEATURES

    final_params = model.named_steps["model"].get_params()

    log_experiment(model_name, mean, std, final_params, cols, print_log)
    print(f"CV {model_name}: mean={mean} std={std} gap={float(train_mean.mean()-mean)}")
    return None

def dnn_cv_with_history(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    scoring: str = "accuracy",
    print_log: bool = False,
):
    """
    Кросс-валидация для dnn с сохранением истории по фолдам.

    Возвращает:
      histories: список history_ по каждому фолду
      fold_scores: список val-score по фолдам
      mean_score: средний score по фолдам
    """
    assert scoring == "accuracy", "Пока считаем только accuracy для истории"

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    histories: list[dict[str, list[float]]] = []
    fold_scores: list[float] = []

    # будем логировать параметры один раз — по первому фолду
    cols = X.columns.values.tolist()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = build_model("dnn", X_tr, params, transform_off=False)

        feat = model.named_steps["feat"]
        prep = model.named_steps["prep"]
        clf = model.named_steps["model"]

        X_tr_fe = feat.transform(X_tr)
        X_val_fe = feat.transform(X_val)

        X_tr_proc = prep.fit_transform(X_tr_fe)
        X_val_proc = prep.transform(X_val_fe)

        clf.fit(X_tr_proc, y_tr, X_val_proc, y_val)

        histories.append(clf.history_)
        y_pred = clf.predict(X_val_proc)

        if scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError("Поддерживается только accuracy в этой обертке")

        fold_scores.append(score)

        if print_log:
            print(f"[fold {fold_idx}] val_{scoring}={score:.4f}")

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    # лог в тот же файл, что и quick_experiment
    # берём итоговые параметры модели с последнего фолда
    final_params = model.named_steps["model"].get_params()
    log_experiment(
        "dnn",
        mean_score,
        std_score,
        final_params,
        cols,
        print_log,
    )

    if print_log:
        print(f"DNN CV: mean={mean_score:.4f} std={std_score:.4f}")

    return histories, fold_scores, mean_score



if __name__ == "__main__":
    run_experiment()