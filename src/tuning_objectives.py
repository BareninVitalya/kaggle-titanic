from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class LogregObjective:
    def __init__(self, base_model, X, y, cv, step_name=None):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv

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
            model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()


class KNNObjective:
    def __init__(self, base_model, X, y, cv, step_name=None):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv

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
            model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
        )

        return scores.mean()

class TreeObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring="accuracy"):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
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

        return scores.mean()


class RFObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring="accuracy"):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
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

        return scores.mean()

class CatBoostObjective:
    def __init__(self, base_model, X, y, cv, step_name=None, scoring="accuracy"):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.cv = cv
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

        return scores.mean()