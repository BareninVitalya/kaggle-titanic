from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from .logging_utils import log_experiment


class LogregObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring: str = 'accuracy',
        log_trials: bool = False,
        logfile: str = "optuna.log"
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring

        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            f"{self.step_name}__C": trial.suggest_float("model__C", 0.01, 8.0, log=True),
            f"{self.step_name}__penalty": "l2",
            f"{self.step_name}__solver": trial.suggest_categorical(
                "model__solver", ["lbfgs"]
            ),
            f"{self.step_name}__l1_ratio": 0.0,
        }
        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=-1
        )
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "logreg", self.logfile)

        return scores.mean()


class KNNObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring: str = 'accuracy',
                 log_trials: bool = False,
                 logfile: str = "optuna.log"
                 ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring

        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            f"{self.step_name}__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 31, step=2),
            f"{self.step_name}__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"]),
            f"{self.step_name}__p": trial.suggest_categorical("model__p", [1, 2]),
            f"{self.step_name}__leaf_size": trial.suggest_int("model__leaf_size", 10, 60),
        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=-1
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "knn", self.logfile)

        return scores.mean()

class TreeObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring: str = 'accuracy',
        log_trials: bool = False,
        logfile: str = "optuna.log"
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring


        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            f"{self.step_name}__criterion": trial.suggest_categorical("model__criterion", ["gini", "entropy", "log_loss"]),
            f"{self.step_name}__max_depth": trial.suggest_int("model__max_depth", 2, 10),
            f"{self.step_name}__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 40),
            f"{self.step_name}__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
            f"{self.step_name}__max_features": trial.suggest_categorical(
                "model__max_features",
                [None, "sqrt", "log2"]
            ),
            f"{self.step_name}__class_weight": trial.suggest_categorical(
                "model__class_weight",
                [None, "balanced"]
            ),
        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "tree", self.logfile)

        return scores.mean()


class RFObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring: str = 'accuracy',
        log_trials: bool = False,
        logfile: str = "optuna.log"
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring

        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            f"{self.step_name}__n_estimators": trial.suggest_int("model__n_estimators", 100, 700, step=100),
            f"{self.step_name}__criterion": trial.suggest_categorical( "model__criterion",
                ["gini", "entropy", "log_loss"]
            ),
            f"{self.step_name}__max_depth": trial.suggest_int("model__max_depth", 2, 12),
            f"{self.step_name}__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 40),
            f"{self.step_name}__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),

        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "tf", self.logfile)

        return scores.mean()

class CatBoostObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring: str = 'accuracy',
        log_trials: bool = False,
        logfile: str = "optuna.log"
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring
        self.step_name = step_name or base_model.steps[-1][0]

    def __call__(self, trial):
        step = self.step_name

        params = {
            f"{step}__iterations": trial.suggest_int(
                "model__iterations", 100, 700, step=100
            ),
            f"{step}__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.2, log=True
            ),
            f"{step}__depth": trial.suggest_int(
                "model__depth", 2, 6
            ),
            f"{step}__l2_leaf_reg": trial.suggest_float(
                "model__l2_leaf_reg", 1.0, 10.0, log=True
            ),
        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "tf", self.logfile)

        return scores.mean()

class XGBObjective:
    def __init__(
        self,
        base_model,
        X,
        y,
        cv,
        step_name=None,
        scoring: str = "accuracy",
        log_trials: bool = False,
        logfile: str = "optuna.log",
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring

        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            f"{self.step_name}__n_estimators": trial.suggest_int(
                "model__n_estimators", 100, 700, step=100
            ),
            f"{self.step_name}__max_depth": trial.suggest_int(
                "model__max_depth", 2, 8
            ),
            f"{self.step_name}__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.3, log=True
            ),
            f"{self.step_name}__min_child_weight": trial.suggest_int(
                "model__min_child_weight", 1, 10
            ),
            f"{self.step_name}__subsample": trial.suggest_float(
                "model__subsample", 0.6, 1.0
            ),
            f"{self.step_name}__colsample_bytree": trial.suggest_float(
                "model__colsample_bytree", 0.6, 1.0
            ),
            f"{self.step_name}__reg_alpha": trial.suggest_float(
                "model__reg_alpha", 1e-3, 10.0, log=True
            ),
            f"{self.step_name}__reg_lambda": trial.suggest_float(
                "model__reg_lambda", 1e-3, 10.0, log=True
            ),
        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "xgb", self.logfile)

        return scores.mean()

class LGBMObjective:
    def __init__(
        self,
        base_model,
        X,
        y,
        cv,
        step_name=None,
        scoring: str = "accuracy",
        log_trials: bool = False,
        logfile: str = "optuna.log",
    ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
        self.log_trials = log_trials
        self.logfile = logfile
        self.scoring = scoring

        if step_name is None:
            self.step_name = base_model.steps[-1][0]
        else:
            self.step_name = step_name

    def __call__(self, trial):
        params = {
            # число деревьев (итерации бустинга)
            f"{self.step_name}__n_estimators": trial.suggest_int(
                "model__n_estimators", 100, 700, step=100
            ),
            # сложность дерева: num_leaves + max_depth
            f"{self.step_name}__num_leaves": trial.suggest_int(
                "model__num_leaves", 15, 63
            ),
            f"{self.step_name}__max_depth": trial.suggest_int(
                "model__max_depth", -1, 10
            ),
            # скорость обучения
            f"{self.step_name}__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.3, log=True
            ),
            # регуляризация структуры дерева
            f"{self.step_name}__min_child_samples": trial.suggest_int(
                "model__min_child_samples", 5, 40
            ),
            # семплинг объектов и признаков
            f"{self.step_name}__subsample": trial.suggest_float(
                "model__subsample", 0.6, 1.0
            ),
            f"{self.step_name}__colsample_bytree": trial.suggest_float(
                "model__colsample_bytree", 0.6, 1.0
            ),
            # L1/L2‑регуляризация листьев
            f"{self.step_name}__reg_alpha": trial.suggest_float(
                "model__reg_alpha", 1e-3, 10.0, log=True
            ),
            f"{self.step_name}__reg_lambda": trial.suggest_float(
                "model__reg_lambda", 1e-3, 10.0, log=True
            ),
        }

        model = clone(self.base_model)
        model.set_params(**params)

        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        if self.log_trials:
            _log_optuna_trial(trial, scores, self.X, "lgbm", self.logfile)

        return scores.mean()


def _log_optuna_trial(
    trial,
    scores,
    X,
    model_name: str,
    logfile: str = "optuna.log",
):
    log_experiment(
        name=f"optuna_{model_name}_trial_{trial.number}",
        score=float(scores.mean()),
        std=float(scores.std()),
        params=trial.params,
        col_names=X.columns.tolist() if hasattr(X, "columns") else [],
        logfile=logfile,
        print_log=False,
    )