from sklearn.model_selection import train_test_split

from src.data import load_train, load_test, save_submission
from src.config import (
    TARGET_COL,
    N_SPLITS,
    SEED,
    DEFAULT_DNN_PARAMS,
    DEFAULT_RF_PARAMS,
    DEFAULT_LOGREG_PARAMS
)
from src.evaluate import cv_scores
from src.ensemble import SoftEnsemble


def main():
    df = load_train()
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model_names = ["logreg", "rf", "dnn"]

    model_params = {
        "logreg": DEFAULT_LOGREG_PARAMS,
        "rf": DEFAULT_RF_PARAMS,
        "dnn": DEFAULT_DNN_PARAMS,
    }

    ensemble = SoftEnsemble(
        model_names=model_names,
        model_params=model_params,
        weights={
            "logreg": 1.0,
            "rf": 1.0,
            "dnn": 1.0,
        },
        threshold=0.5,
        transform_off=False,
    )

    val_mean, val_std, val_scores, train_data = cv_scores(
        model=ensemble,
        X=X_raw,
        y=y,
        n_splits=N_SPLITS,
        seed=SEED,
        scoring="accuracy",
        return_train_score=False,
    )
    print(
        f"CV SoftEnsemble: "
        f"mean={val_mean} "
        f"std={val_std} "
        f"gap={float(train_data - val_std)}"
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print(f'Fit and predict ...')
    ensemble.fit_build(X_tr, y_tr)
    df_submission = ensemble.predict_test(load_test())

    save_submission(df_submission)


if __name__ == "__main__":
    main()