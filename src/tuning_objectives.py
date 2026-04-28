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
            f"{self.step_name}__C": trial.suggest_float("model__C", 1e-3, 1e3, log=True),
            f"{self.step_name}__solver": trial.suggest_categorical(
                "model__solver", ["lbfgs", "liblinear"]
            ),
        }
        model = clone(self.base_model)
        model.set_params(**params)
        scores = cross_val_score(
            model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()