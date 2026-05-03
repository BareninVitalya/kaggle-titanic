from pathlib import Path
from scipy.stats import loguniform
from .tuning_objectives import LogregObjective

# Корень проекта: kaggle-titanic/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

SEED = 42
TARGET_COL = "Survived"

N_SPLITS = 5

# Фичи, которые в ноутбуке показали себя "шумными" и были выкинуты [file:1]
NOISE_FEATURES = ["TicketPrefix","Fare_bin", "Pclass"]

# Позже мы ещё удаляли SibSp и Parch без потери качества [file:1]
DROP_SIBSP_PARCH = True

# ── Дефолтные параметры моделей (из экспериментов в ноутбуке) ─────────────────
DEFAULT_LOGREG_PARAMS = {
    "max_iter":    2000,
    "C":           1.0,
    "l1_ratio":    0.0,
    "solver":      "lbfgs",
    "random_state": SEED,
}

DEFAULT_RF_PARAMS = {
    "n_estimators":    300,
    "max_depth":       5,
    "min_samples_split": 10,
    "random_state":    SEED,
}

DEFAULT_GB_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.03,
    "max_depth":       2,
    "min_samples_leaf": 3,
    "random_state":    SEED,
}

DEFAULT_KNN_PARAMS = {
    "n_neighbors": 14,
    "weights":     "distance",
}

DEFAULT_TREE_PARAMS = {
    "max_depth":        7,
    "min_samples_leaf": 5,
    "random_state":     SEED,
}





RANDOM_SEARCH_SPACE = {
    "logreg": {
        "model__C": loguniform(1e-3, 1e3),
        "model__solver": ["lbfgs", "liblinear"],
    },
    # сюда потом можно добавить "rf", "xgb" и т.д.
}

# def optuna_space_logreg(trial):
#     return {
#         "model__C": trial.suggest_float("model__C", 1e-3, 1e3, log=True),
#         "model__solver": trial.suggest_categorical(
#             "model__solver", ["lbfgs", "liblinear"]
#         ),
#     }

# OPTUNA_OBJECTIVES = {
#     "logreg": LogregObjective,
#     # "rf": RandomForestObjective,  # в будущем
# }

# ── OpenFE ────────────────────────────────────────────────────────────────────
USE_OPENFE = False

OPENFE_PARAMS = {
    "n_features": 30,
    "corr_threshold": 0.95,
    "greedy_threshold": 0.001,
    "use_ablation": True,
    "ablation_step": 5,
    "n_jobs": 1,
}

NUMERIC_AS_CATEGORICAL_MAX_UNIQUE = 10