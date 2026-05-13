"""Утилиты для ансамблирования моделей: soft voting, VotingClassifier и stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from . import modeling
from .config import SEED, N_SPLITS

from pathlib import Path


def _build_submission(
    predictions: np.ndarray,
    ids: pd.Series,
) -> pd.DataFrame:
    """Собрать submission DataFrame из PassengerId и предсказаний."""
    return pd.DataFrame(
        {
            "PassengerId": ids.squeeze().values,
            "Survived": predictions.astype(int),
        }
    )


class SoftEnsemble:
    """Кастомный soft-voting ансамбль с поддержкой весов и порога бинаризации."""

    def __init__(
            self,
            model_names: list[str],
            model_params: dict[str, dict] | None = None,
            weights: dict[str, float] | None = None,
            threshold: float = 0.5,
            transform_off: bool = False,
    ) -> None:
        """Инициализировать ансамбль моделей и его параметры."""
        self.model_names = model_names
        self.model_params = model_params or {}
        self.weights = weights or {name: 1.0 for name in model_names}
        self.threshold = threshold
        self.transform_off = transform_off

        self.models_ = {}
        self.is_fitted_ = False

    def build(self, X: pd.DataFrame) -> "SoftEnsemble":
        """Собрать базовые модели ансамбля по их именам и параметрам."""
        self.models_ = {}

        for name in self.model_names:
            params = self.model_params.get(name)
            model = modeling.build_model(
                name,
                X,
                params=params,
                transform_off=self.transform_off,
            )
            self.models_[name] = model

        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SoftEnsemble":
        """Обучить все модели ансамбля на переданных данных."""
        if not self.models_:
            self.build(X)

        for name, model in self.models_.items():
            model.fit(X, y)

        self.is_fitted_ = True
        return self

    def fit_build(self, X: pd.DataFrame, y: pd.Series) -> "SoftEnsemble":
        """Собрать и обучить ансамбль за один вызов."""
        self.build(X)
        self.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Вернуть усреднённые вероятности положительного класса по всем моделям ансамбля."""

        if not self.is_fitted_:
            raise ValueError("Ensemble is not fitted yet")

        total_weight = sum(self.weights[name] for name in self.model_names)
        ensemble_proba = np.zeros(len(X), dtype=float)

        for name in self.model_names:
            model = self.models_[name]
            proba = model.predict_proba(X)[:, 1]
            ensemble_proba += self.weights[name] * proba

        ensemble_proba /= total_weight
        return ensemble_proba

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Преобразовать вероятности ансамбля в бинарные предсказания по заданному порогу."""
        threshold = self.threshold if threshold is None else threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_test(
            self,
            df_test: pd.DataFrame,
            ids: pd.DataFrame | None = None,
            file_name: str | None = None,
    ) -> pd.DataFrame:
        """Сформировать submission DataFrame для тестового набора."""

        pred = self.predict(df_test)

        if ids is None:
            ids = df_test[["PassengerId"]]

        submission = _build_submission(pred, ids)

        if file_name is not None:
            if not file_name.endswith(".csv"):
                file_name += ".csv"
            submission.to_csv(file_name, index=False)
            print(f"Saved to {file_name}")

        return submission


class VotingEnsemble:
    """Обёртка над sklearn VotingClassifier для сборки, обучения и инференса ансамбля."""

    def __init__(
        self,
        model_names: list[str],
        model_params: dict[str, dict] | None = None,
        voting: str = "soft",
        weights: list[float] | None = None,
        transform_off: bool = True,
        flatten_transform: bool = True,
    ) -> None:
        """Инициализировать конфигурацию voting-ансамбля."""
        self.model_names = model_names
        self.model_params = model_params or {}
        self.voting = voting
        self.weights = weights
        self.transform_off = transform_off
        self.flatten_transform = flatten_transform

        self.ensemble_: VotingClassifier | None = None

    def build(self, X: pd.DataFrame) -> "VotingEnsemble":
        """Собрать VotingClassifier из списка базовых моделей."""
        estimators: list[tuple[str, object]] = []

        for name in self.model_names:
            params = self.model_params.get(name)
            model = modeling.build_model(
                name,
                X,
                params=params,
                transform_off=self.transform_off,
            )
            estimators.append((name, model))

        self.ensemble_ = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            flatten_transform=self.flatten_transform,
        )
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VotingEnsemble":
        """Собрать и обучить voting-ансамбль на переданных данных."""
        if self.ensemble_ is None:
            self.build(X)

        self.ensemble_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Вернуть бинарные предсказания обученного voting-ансамбля."""
        if self.ensemble_ is None:
            raise ValueError("VotingEnsemble is not fitted yet.")

        return self.ensemble_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Вернуть вероятности положительного класса для voting-ансамбля."""
        if self.ensemble_ is None:
            raise ValueError("VotingEnsemble is not fitted yet.")

        if self.voting != "soft":
            raise ValueError("predict_proba is available only when voting='soft'.")

        return self.ensemble_.predict_proba(X)[:, 1]

    def predict_test(
        self,
        df_test: pd.DataFrame,
        ids: pd.DataFrame | None = None,
        file_name: str | Path | None = None,
    ) -> pd.DataFrame:
        """Сформировать submission DataFrame для тестового набора."""
        predictions = self.predict(df_test)

        if ids is None:
            ids = df_test[["PassengerId"]]

        submission = _build_submission(predictions, ids)

        if file_name is not None:
            output_path = Path(file_name)
            if output_path.suffix != ".csv":
                output_path = output_path.with_suffix(".csv")
            submission.to_csv(output_path, index=False)

        return submission


class StackingEnsemble:
    """Кастомная обёртка для обучения stacking-ансамбля с OOF-признаками."""

    def __init__(
        self,
        model_names: list[str],
        model_params: dict[str, dict] | None = None,
        meta_model_name: str = "ridge",
        meta_params: dict | None = None,
        n_splits: int = N_SPLITS,
        random_state: int = SEED,
        transform_off: bool = False,
        threshold: float = 0.5,
    ) -> None:
        """Инициализировать конфигурацию stacking-ансамбля."""
        self.model_names = model_names
        self.model_params = model_params or {}
        self.meta_model_name = meta_model_name
        self.meta_params = meta_params or {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.transform_off = transform_off
        self.threshold = threshold

        self.meta_model_ = None
        self.oof_pred_: pd.DataFrame | None = None
        self.test_pred_: pd.DataFrame | None = None
        self.base_models_: dict[str, list] = {}

    def _make_meta_model(self):
        """Создать мета-модель по её имени и параметрам."""
        if self.meta_model_name == "ridge":
            return RidgeClassifier(**self.meta_params)
        if self.meta_model_name == "logreg":
            return LogisticRegression(**self.meta_params)

        raise ValueError(f"Unknown meta_model: {self.meta_model_name}")

    def build_oof_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Построить OOF-признаки для обучения мета-модели и признаки для теста."""
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        oof_pred = pd.DataFrame(index=X.index)
        test_pred = pd.DataFrame(index=X_test.index)

        self.base_models_ = {}

        for model_name in self.model_names:
            params = self.model_params.get(model_name)
            self.base_models_[model_name] = []

            oof_col = np.zeros(len(X))
            test_fold_pred = np.zeros((len(X_test), self.n_splits))

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
                X_tr = X.iloc[tr_idx]
                y_tr = y.iloc[tr_idx]
                X_val = X.iloc[val_idx]

                model = modeling.build_model(
                    model_name,
                    X_tr,
                    params=params,
                    transform_off=self.transform_off,
                )

                model.fit(X_tr, y_tr)

                self.base_models_[model_name].append(model)

                val_proba = model.predict_proba(X_val)[:, 1]
                test_proba = model.predict_proba(X_test)[:, 1]

                oof_col[val_idx] = val_proba
                test_fold_pred[:, fold - 1] = test_proba

            oof_pred[model_name] = oof_col
            test_pred[model_name] = test_fold_pred.mean(axis=1)

        self.oof_pred_ = oof_pred
        self.test_pred_ = test_pred
        return oof_pred, test_pred

    def build_meta_data(
            self,
            X_new: pd.DataFrame,
    ) -> pd.DataFrame:
        """Построить мета-признаки для новых данных через сохранённые fold-модели."""
        if not self.base_models_:
            raise ValueError("Base models are not fitted yet. Call fit() first.")

        meta_pred = pd.DataFrame(index=X_new.index)

        for model_name in self.model_names:
            if model_name not in self.base_models_:
                raise ValueError(f"Models for '{model_name}' are not available.")

            fold_models = self.base_models_[model_name]
            fold_pred = np.zeros((len(X_new), len(fold_models)))

            for fold, model in enumerate(fold_models):
                fold_pred[:, fold] = model.predict_proba(X_new)[:, 1]

            meta_pred[model_name] = fold_pred.mean(axis=1)

        return meta_pred

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
    ) -> "StackingEnsemble":
        """Построить OOF-признаки и обучить мета-модель stacking-ансамбля."""
        oof_pred, test_pred = self.build_oof_data(X=X, y=y, X_test=X_test)

        self.meta_model_ = self._make_meta_model()
        self.meta_model_.fit(oof_pred, y)
        self.test_pred_ = test_pred

        return self

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float | None = None,
        is_meta_data: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть вероятности и бинарные предсказания мета-модели."""
        if self.meta_model_ is None:
            raise ValueError("StackingEnsemble is not fitted yet.")

        X_meta = X if is_meta_data else self.build_meta_data(X)

        decision_threshold = self.threshold if threshold is None else threshold

        if hasattr(self.meta_model_, "predict_proba"):
            proba = self.meta_model_.predict_proba(X_meta)[:, 1]
            pred = (proba >= decision_threshold).astype(int)
            return proba, pred

        raw_pred = self.meta_model_.predict(X_meta)

        if getattr(raw_pred, "ndim", 1) > 1:
            raw_pred = raw_pred.ravel()

        proba = np.clip(raw_pred, 0, 1)
        pred = (proba >= decision_threshold).astype(int)
        return proba, pred

    def predict_test(
        self,
        df_test: pd.DataFrame,
        X_meta_test: pd.DataFrame | None = None,
        file_name: str | Path | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Сформировать submission DataFrame для тестового набора."""
        meta_features = X_meta_test if X_meta_test is not None else self.test_pred_

        if meta_features is None:
            raise ValueError("Test meta-features are not available. Fit the ensemble first or pass X_meta_test.")

        _, pred = self.predict(meta_features, threshold=threshold)
        ids = df_test[["PassengerId"]]
        submission = _build_submission(pred, ids)

        if file_name is not None:
            output_path = Path(file_name)
            if output_path.suffix != ".csv":
                output_path = output_path.with_suffix(".csv")
            submission.to_csv(output_path, index=False)

        return submission