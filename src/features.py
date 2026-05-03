import pandas as pd
import numpy as np
from .config import NOISE_FEATURES, DROP_SIBSP_PARCH
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatures:
    """Feature engineering для датасета Titanic."""

    def __init__(
            self, drop_noise: bool = True,
            drop_sibsp_parch: bool = DROP_SIBSP_PARCH,
            use_log_fare: bool = True,
            use_age_bins: bool = True,
            use_fare_bins: bool = True,
            use_pclass_sex: bool = True,
            model_type: str = "linear"
    ):
        self.drop_noise = drop_noise
        self.drop_sibsp_parch = drop_sibsp_parch
        self.use_log_fare = use_log_fare
        self.use_age_bins = use_age_bins
        self.use_fare_bins = use_fare_bins
        self.use_pclass_sex = use_pclass_sex
        self.model_type = model_type

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        # ", Title." между фамилией и именем [file:1]
        titles = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

        # Нормализация
        titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        rare_titles = ["Dr", "Rev", "Major", "Col", "Jonkheer", "Lady", "Capt", "Don", "Sir"]
        titles = titles.replace(rare_titles, "Rare")

        df["Title"] = titles
        return df

    def _add_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        df["familysize"] = df["SibSp"] + df["Parch"] + 1
        df["isalone"] = (df["familysize"] == 1).astype(int)
        return df

    def _add_ticket_group_size(self, df: pd.DataFrame) -> pd.DataFrame:
        # Сколько людей с таким же билетом [file:1]
        df["ticketgroupsize"] = df.groupby("Ticket")["Ticket"].transform("count")
        return df

    def _add_ticket_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        # Убираем цифры и разделители, оставляем префикс или "NONE" [file:1]
        s = df["Ticket"].astype(str)
        s = s.str.replace(r"[\./]", " ", regex=True)
        s = s.str.replace(r"\d+", "", regex=True)
        s = s.str.strip()
        s = s.replace("", "NONE")
        df["TicketPrefix"] = s
        return df

    def _add_cabin_deck(self, df: pd.DataFrame) -> pd.DataFrame:
        # Первая буква кабины, NaN → "Unknown" [file:1]
        deck = df["Cabin"].astype(str).str[0]
        deck = deck.replace("n", "Unknown")  # "nan" → "Unknown"
        df["CabinDeck"] = deck
        return df

    def _encode_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
        return df

    def _add_pclass_sex_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Pclass_Sex"] =df["Pclass"] * df["Sex"]
        return df

    def _add_log_fare_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Логарифмирует Fare для уменьшения асимметрии распределения."""
        df["Log_Fare"] = np.log1p(df["Fare"])
        return df

    def _bin_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Разбиваем Age на группы"""
        df["Age_bin"] = pd.cut(df["Age"], bins=[0, 12, 18, 40, 60, 100], labels=False)
        return df

    def _bin_fare_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Разбиваем Fare на группы"""
        df["Fare_bin"] = pd.qcut(df["Fare"], q=4, labels=False)
        return df

    def _transform_fare_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Логарифмирует Fare на месте для уменьшения асимметрии."""
        df["Fare"] = np.log1p(df["Fare"])
        return df

    def _drop_noise_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        for col in features:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        return df

    def _drop_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["Age"], inplace=True)
        return df

    def _drop_fare_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["Fare"], inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Главный метод: принимает сырые колонки Titanic, возвращает X с готовыми фичами,
        максимально повторяя логику ноутбука перед лучшим логрегом (~0.83). [file:1]
        """
        df = df.copy()

        df = self._extract_title(df)
        df = self._add_family_size(df)
        df = self._add_ticket_group_size(df)
        df = self._add_ticket_prefix(df)
        df = self._add_cabin_deck(df)
        df = self._encode_sex(df)

        if self.use_log_fare:
            df = self._transform_fare_log(df)

        if self.use_age_bins:
            df = self._bin_age_feature(df)
            df = self._drop_age_feature(df)

        if self.use_fare_bins:
            df = self._bin_fare_feature(df)
            df = self._drop_fare_feature(df)

        if self.use_pclass_sex:
            df = self._add_pclass_sex_feature(df)

        df = self._drop_noise_features(df, ["PassengerId", "Name", "Ticket", "Cabin"])
        df = self._drop_noise_features(df, ["SibSp", "Parch"])
        df = self._drop_noise_features(df, ["TicketPrefix","ticketgroupsize", "CabinDeck"])

        return df

class TitanicFeaturesTransformer(TitanicFeatures, BaseEstimator, TransformerMixin):
    """Sklearn-совместимый wrapper над TitanicFeatures."""

    def fit(self, X, y=None):
        # Stateful-логики пока нет, просто возвращаем self
        return self