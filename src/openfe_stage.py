from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS_DIR, SEED
from .evaluate import cv_scores
from .feature_search import apply_openfe, run_openfe
from .modeling import build_model
from openfe import tree_to_formula


@dataclass
class OpenFEStage:
    """Отдельный этап поиска, отбора и применения OpenFE-признаков."""

    enabled: bool = True
    n_features: int = 30
    model_name: str = "logreg"
    # --- фильтр мультиколлинеарности (применяется до greedy) ---
    corr_threshold: float = 0.95
    # --- greedy selection ---
    greedy_threshold: float = 0.002
    # --- ablation (batch, по шагам) ---
    use_ablation: bool = True
    ablation_step: int = 5
    # --- прочее ---
    n_jobs: int = 1
    seed: int = SEED

    # fitted artifacts
    features_: list = field(default_factory=list)
    selected_columns_: list[str] = field(default_factory=list)
    base_columns_: list[str] = field(default_factory=list)
    greedy_history_: list[dict] = field(default_factory=list)
    dropped_corr_: list[dict] = field(default_factory=list)
    ablation_report_: pd.DataFrame | None = None
    selected_feature_registry_: pd.DataFrame | None = None

    def _drop_correlated(
        self,
        X_full: pd.DataFrame,
        ofe_cols: list[str],
    ) -> list[str]:
        """
        Убирает OFE-кандидаты, у которых |r| >= corr_threshold
        с любым из базовых признаков.
        """
        self.dropped_corr_ = []

        num_cols = X_full.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return ofe_cols

        corr = X_full[num_cols].corr().abs()

        base_num = [c for c in self.base_columns_ if c in num_cols]
        ofe_num  = [c for c in ofe_cols if c in num_cols]

        kept: list[str] = []
        for col in ofe_cols:
            if col not in ofe_num:
                kept.append(col)
                continue

            max_corr_val = corr.loc[col, base_num].max() if base_num else 0.0
            max_corr_col = corr.loc[col, base_num].idxmax() if base_num else None

            if max_corr_val >= self.corr_threshold:
                self.dropped_corr_.append({
                    "dropped": col,
                    "correlated_with": max_corr_col,
                    "corr": round(max_corr_val, 4),
                })
                print(
                    f"  CORR DROP {col:30s}  |r|={max_corr_val:.4f}"
                    f"  (with {max_corr_col})"
                )
            else:
                kept.append(col)

        print(f"\nAfter corr filter: {len(ofe_cols)} → {len(kept)} OFE candidates")
        return kept

    def _greedy_select(
        self,
        X_base: pd.DataFrame,
        X_with_ofe: pd.DataFrame,
        y: pd.Series,
        ofe_cols: list[str],
    ) -> list[str]:
        current_X = X_base.copy()
        current_score, _, _ = cv_scores(
            build_model(self.model_name, current_X), current_X, y
        )
        selected: list[str] = []
        self.greedy_history_ = []

        per_feat: list[dict] = []
        for col in ofe_cols:
            cand_X = pd.concat([X_base, X_with_ofe[[col]]], axis=1)
            mean, std, _ = cv_scores(
                build_model(self.model_name, cand_X), cand_X, y
            )
            per_feat.append({
                "col": col, "mean": mean, "std": std,
                "delta": round(mean - current_score, 5),
            })

        sorted_feats = sorted(per_feat, key=lambda r: r["delta"], reverse=True)

        for row in sorted_feats:
            col = row["col"]
            cand_X = pd.concat([current_X, X_with_ofe[[col]]], axis=1)
            mean, std, _ = cv_scores(
                build_model(self.model_name, cand_X), cand_X, y
            )
            delta = mean - current_score
            self.greedy_history_.append({
                "col": col, "mean": round(mean, 5),
                "std": round(std, 5), "delta": round(delta, 5),
            })

            if delta >= self.greedy_threshold:
                print(f"  KEEP {col:30s}  mean={mean:.5f}  delta={delta:+.5f}")
                current_X = cand_X
                selected.append(col)
                current_score = mean
            else:
                print(f"  DROP {col:30s}  mean={mean:.5f}  delta={delta:+.5f}")

        print(f"\nGreedy final score : {current_score:.5f}")
        print(f"Selected features  : {selected}")
        return selected

    def _ablation(
        self,
        X_base: pd.DataFrame,
        X_with_ofe: pd.DataFrame,
        y: pd.Series,
        ofe_cols: list[str],
    ) -> pd.DataFrame:
        base_score, base_std, _ = cv_scores(
            build_model(self.model_name, X_base), X_base, y
        )
        rows = [{"n_features": 0, "mean": base_score,
                 "std": base_std, "delta": 0.0}]

        for n in range(self.ablation_step, len(ofe_cols) + 1, self.ablation_step):
            cols_batch = ofe_cols[:n]
            X_exp = pd.concat([X_base, X_with_ofe[cols_batch]], axis=1)
            mean, std, _ = cv_scores(
                build_model(self.model_name, X_exp), X_exp, y
            )
            rows.append({
                "n_features": n, "mean": mean,
                "std": std, "delta": round(mean - base_score, 5),
            })
            print(f"  n={n:3d}  mean={mean:.4f}  delta={mean - base_score:+.5f}")

        return pd.DataFrame(rows)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OpenFEStage":
        self.base_columns_ = list(X.columns)

        if not self.enabled:
            return self

        # 1. Генерация кандидатов
        self.features_ = run_openfe(
            X_train=X, y_train=y,
            n_features=self.n_features,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )

        # 2. Применяем все кандидаты к train
        X_with_ofe, _ = apply_openfe(X, X.copy(), self.features_)
        ofe_cols = [c for c in X_with_ofe.columns if c not in self.base_columns_]

        # 3. Фильтр мультиколлинеарности (до ablation и greedy)
        print("\n── Correlation filter ──")
        ofe_cols = self._drop_correlated(X_with_ofe, ofe_cols)

        # 4. Batch-ablation (опционально, только по отфильтрованным)
        if self.use_ablation and ofe_cols:
            print("\n── Ablation (batch) ──")
            self.ablation_report_ = self._ablation(X, X_with_ofe, y, ofe_cols)

        # 5. Жадный отбор
        print("\n── Greedy selection ──")
        self.selected_columns_ = self._greedy_select(X, X_with_ofe, y, ofe_cols)

        self.selected_feature_registry_ = self.build_feature_registry(X_with_ofe)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.features_:
            return X.copy()

        X_with_ofe, _ = apply_openfe(X, X.copy(), self.features_)
        keep_cols = list(X.columns) + self.selected_columns_
        keep_cols = [c for c in keep_cols if c in X_with_ofe.columns]
        return X_with_ofe[keep_cols].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        return list(self.base_columns_) + list(self.selected_columns_)

    def get_greedy_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.greedy_history_)

    def get_corr_dropped(self) -> pd.DataFrame:
        return pd.DataFrame(self.dropped_corr_)

    def save(self, path: Path | None = None) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = MODELS_DIR / f"openfe_stage_{self.model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved OpenFEStage to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "OpenFEStage":
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_feature_registry(
            self,
            X_with_ofe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Строит реестр OFE-признаков:
        - техническое имя колонки
        - человекочитаемая формула
        - оператор
        - была ли фича выбрана greedy-отбором
        - метрики greedy и корреляционного фильтра
        """
        all_new_cols = [c for c in X_with_ofe.columns if c not in self.base_columns_]

        rows = []
        greedy_map = {
            row["col"]: row
            for row in self.greedy_history_
        }
        corr_drop_map = {
            row["dropped"]: row
            for row in self.dropped_corr_
        }

        for feat, col in zip(self.features_, all_new_cols):
            try:
                formula = tree_to_formula(feat)
            except Exception:
                formula = None

            row = {
                "column": col,
                "operator": getattr(feat, "name", None),
                "formula": formula,
                "selected": col in self.selected_columns_,
                "greedy_mean": None,
                "greedy_std": None,
                "greedy_delta": None,
                "dropped_by_corr": col in corr_drop_map,
                "corr_with": None,
                "corr_value": None,
            }

            if col in greedy_map:
                row["greedy_mean"] = greedy_map[col].get("mean")
                row["greedy_std"] = greedy_map[col].get("std")
                row["greedy_delta"] = greedy_map[col].get("delta")

            if col in corr_drop_map:
                row["corr_with"] = corr_drop_map[col].get("correlated_with")
                row["corr_value"] = corr_drop_map[col].get("corr")

            rows.append(row)

        df = pd.DataFrame(rows)
        self.selected_feature_registry_ = df.to_dict("records")
        return df