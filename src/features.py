import pandas as pd
import numpy as np
from .config import NOISE_FEATURES, MODELS_DIR, PCLASS_SEX_RANK
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class TitanicFeaturesBase:
    """Общие признаки для обеих задач (Age и Survival)."""

    def _extract_title(self, df):
        titles = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        rare = ["Dr", "Rev", "Major", "Col", "Jonkheer", "Lady",
                "Capt", "Don", "Sir", "the Countess", "Dona"]
        titles = titles.replace(rare, "Rare")
        df["Title"] = titles
        return df

    def _add_family_size(self, df):
        df["familysize"] = df["SibSp"] + df["Parch"] + 1
        df["isalone"] = (df["familysize"] == 1).astype(int)
        return df

    def _add_ticket_group_size(self, df):
        df["ticketgroupsize"] = df.groupby("Ticket")["Ticket"].transform("count")
        return df

    def _add_ticket_prefix(self, df):
        s = df["Ticket"].astype(str)
        s = s.str.replace(r"[\./]", " ", regex=True).str.replace(r"\d+", "", regex=True).str.strip()
        s = s.replace("", "NONE")
        df["TicketPrefix"] = s
        return df

    def _add_cabin_deck(self, df):
        deck = df["Cabin"].astype(str).str[0]
        df["CabinDeck"] = deck.replace("n", "Unknown")
        return df

    def _encode_sex(self, df):
        df["Sex"] = df["Sex"].replace({"male": 0, "female": 1}).astype(int)
        return df

    def _add_pclass_sex_feature(self, df):
        df["Pclass_Sex"] = df["Pclass"].astype(int) * df["Sex"].astype(int)
        return df

    def _transform_fare_log(self, df):
        df["Fare"] = np.log1p(df["Fare"])
        return df

    def _bin_age_feature(self, df):
        df["Age_bin"] = pd.cut(df["Age"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], labels=False)
        return df

    def _bin_fare_log_feature(self, df):
        df["Fare_bin"] = pd.cut(
            df["Fare"],
            bins=[0, 2.2, 2.6, 3.0, 3.4, 4.0, np.inf],
            labels=False, include_lowest=True
        )
        return df

    def _bin_fare_feature(self, df):
        df["Fare_bin"] = pd.cut(
            df["Fare"],
            bins=[-np.inf, 7.90, 14.45, 31.28, 120.0, np.inf],
            labels=False, include_lowest=True
        )
        return df

    def _drop(self, df, columns):
        return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")

    def _base_transform(self, df):
        """Шаги, общие для обеих задач."""
        df = df.copy()
        df = self._extract_title(df)
        df = self._add_family_size(df)
        df = self._add_ticket_group_size(df)
        df = self._add_ticket_prefix(df)
        df = self._add_cabin_deck(df)
        df = self._encode_sex(df)
        return df


# ──────────────────────────────────────────────
# Класс для задачи регрессии Age
# ──────────────────────────────────────────────

class AgeFeatures(TitanicFeaturesBase):
    """
    Препроцессинг для обучения модели предсказания возраста.
    Не использует Age и Age_bin как фичи, не дропает PassengerId
    (удобно для отладки). Таргет = Age.
    """

    def __init__(self, use_log_fare: bool = True, is_fit: bool = False):
        self.use_log_fare = use_log_fare
        self.is_fit = is_fit

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._base_transform(df)
        if not self.is_fit:
            df = self._bin_age_feature(df)
        df = self._transform_fare_log(df)


        # Дропаем то, что нельзя использовать для предсказания Age
        df = self._drop(df, [
            "PassengerId", "Name", "Ticket", "Cabin",
            "Survived",               # если есть
            "Age",                    # таргет — не фича
            "TicketPrefix", "ticketgroupsize", "CabinDeck", "familysize",
        ])
        return df


# ──────────────────────────────────────────────
# Класс для задачи классификации Survived
# ──────────────────────────────────────────────

class SurvivalFeatures(TitanicFeaturesBase):
    """
    Препроцессинг для основной задачи: предсказание выживания.
    Использует внутреннюю модель (age_model) для заполнения Age.
    """

    def __init__(
        self,
        drop_sibsp_parch: bool = True,
        use_log_fare: bool = True,
        use_age_bins: bool = True,
        use_fare_bins: bool = True,
        use_pclass_sex: bool = True,
        age_model_path: str = rf"{MODELS_DIR}\age_model.joblib",
    ):
        self.drop_sibsp_parch = drop_sibsp_parch
        self.use_log_fare = use_log_fare
        self.use_age_bins = use_age_bins
        self.use_fare_bins = use_fare_bins
        self.use_pclass_sex = use_pclass_sex
        self.age_model_path = age_model_path

    def _load_age_model(self):
        return joblib.load(self.age_model_path)

    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет Age пропуски через отдельную age-модель.
        Важно: age-модель использует AgeFeatures.transform() — без Age,
        поэтому нет рекурсии и нет утечки.
        """
        mask = df["Age"].isna()
        if mask.sum() == 0:
            return df

        age_model = self._load_age_model()

        # Подготовка фич через AgeFeatures — отдельный пайплайн
        age_feat = AgeFeatures(
            use_log_fare=self.use_log_fare
        )
        X_unknown = age_feat.transform(df.loc[mask])
        df.loc[mask, "Age"] = age_model.predict(X_unknown)
        return df

    def _impute_age_bin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предсказывает Age_bin для пропущенных, если use_age_bins=True."""
        mask = df["Age"].isna()
        if mask.sum() == 0:
            return df

        age_model = self._load_age_model()
        X_unknown = df.loc[mask].drop(columns=["Age", "Age_bin"], errors="ignore")
        pred = np.rint(age_model.predict(X_unknown)).clip(0, 4).astype(int)
        df.loc[mask, "Age_bin"] = pred
        return df

    def _impute_age_median_of_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняем Age медианой по группе Title.
        """
        df = df.copy()

        df["Age"] = (
            df
            .groupby("Title")["Age"]
            .transform(lambda x: x.fillna(x.median()))
        )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._base_transform(df)
        df = self._impute_age_median_of_title(df)

        if self.use_age_bins:
            df = self._bin_age_feature(df)   # считаем Age_bin из известных
            df = self._drop(df, ["Age"])
        else:
            df = self._bin_age_feature(df)
            df = self._drop(df, ["Age_bin"])

        if self.use_log_fare:
            df = self._transform_fare_log(df)

        if self.use_fare_bins:
            if self.use_log_fare:
                df = self._bin_fare_log_feature(df)
            else:
                df = self._bin_fare_feature(df)
            df = self._drop(df, ["Fare"])

        if self.use_pclass_sex:
            df = self._add_pclass_sex_feature(df)

        cols = set(NOISE_FEATURES.copy())

        if not self.drop_sibsp_parch:
            cols -= {"SibSp", "Parch"}

        df = self._drop(df, cols)

        return df

# ──────────────────────────────────────────────
# Sklearn-совместимые трансформеры
# ──────────────────────────────────────────────

class TitanicTransformerMixin(BaseEstimator, TransformerMixin):
    """Добавляет fit() — stateful-логики нет, просто возвращает self."""

    def fit(self, X, y=None):
        return self

class AgeTransformer(AgeFeatures, TitanicTransformerMixin):
    """Sklearn Pipeline-совместимый трансформер для задачи регрессии Age."""
    pass

class SurvivalTransformer(SurvivalFeatures, TitanicTransformerMixin):
    """Sklearn Pipeline-совместимый трансформер для задачи Survived."""
    pass
