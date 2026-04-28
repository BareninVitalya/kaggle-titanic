from pathlib import Path
import pandas as pd

from .config import TRAIN_PATH, TEST_PATH, PROCESSED_DATA_DIR


def load_train() -> pd.DataFrame:
    """Загрузить исходный train.csv."""
    return pd.read_csv(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    """Загрузить исходный test.csv."""
    return pd.read_csv(TEST_PATH)


def save_processed(df: pd.DataFrame, name: str = "train_clean.csv") -> Path:
    """Сохранить обработанный датафрейм в data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
    return path