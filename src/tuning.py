# tuning.py
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
from scipy.stats import loguniform
from .tuning_objectives import LogregObjective, KNNObjective, TreeObjective, RFObjective, CatBoostObjective
import optuna

from .modeling import build_model
from .config import RANDOM_SEARCH_SPACE, SEED

def tune_with_random_search(
    model_name: str,
    X,
    y,
    base_params: dict | None = None,
    n_iter: int = 30,
    scoring: str = "accuracy",
    random_state: int = SEED,
):
    base_model = build_model(model_name, X, params=base_params, transform_off=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    param_distributions = RANDOM_SEARCH_SPACE[model_name]

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    return best_model, best_params, search

def tune_with_optuna(
    model_name: str,
    X,
    y,
    base_params: dict | None = None,
    n_trials: int = 50,
    scoring: str = "accuracy",
    random_state: int = SEED,
    transform_off: bool = True
):
    if model_name == 'knn':
        objective_class = KNNObjective
    elif model_name == "tree":
        objective_class = TreeObjective
    elif model_name in ("rf", "random_forest", "randomforest"):
        objective_class = RFObjective
    elif model_name in ("catboost", "cat"):
        objective_class = CatBoostObjective
    # elif model_name == 'logreg':
    else:
        base_params = base_params or {}
        base_params["penalty"] = "l2" # костыль который нужен чтобы работало на версии 1.8 sklearn
        objective_class = LogregObjective


    base_model = build_model(model_name, X, params=base_params, transform_off=transform_off)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    objective = objective_class(base_model, X, y, cv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_model = clone(base_model)

    best_model.set_params(**study.best_trial.params)
    best_model.fit(X, y)
    return best_model, study