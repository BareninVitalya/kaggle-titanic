from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import json

import pandas as pd

from .config import LOG_DIR


def log_experiment(
    name: str,
    score: float,
    std: float,
    params: Optional[Dict[str, Any]] = None,
    col_names: list = None,
    print_log: bool = False,
    logfile: str = "experiments.log",
) -> Path:
    """Записать результаты эксперимента в текстовый лог и вывести строку на экран."""
    if params is None:
        params = {}

    if col_names is None:
        col_names = list()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / logfile

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": ts,
        "name": name,
        "score": round(score, 6),
        "std": round(std, 6),
        "params": params,
        "col_names": col_names,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)

    if print_log:
        print(f"{ts} {name} score={score:.6f} std={std:.6f} col_names={col_names} params={params}")

    return path


def load_experiments(
    logfile: str = "experiments.log",
) -> pd.DataFrame:
    """Загрузить лог экспериментов как DataFrame (если файла нет — пустой DF)."""
    columns = ["timestamp", "name", "score", "std", "params", "col_names"]
    path = LOG_DIR / logfile

    if not path.exists():
        return pd.DataFrame(columns=columns)

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rows.append(record)
            except json.JSONDecodeError:
                # Поддержка старого формата строк (обратная совместимость)
                continue

    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["score"] = df["score"].astype(float)
    df["std"] = df["std"].astype(float)
    return df