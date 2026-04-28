from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .config import LOG_DIR


def log_experiment(
    name: str,
    score: float,
    std: float,
    params: Optional[Dict[str, Any]] = None,
    logfile: str = "experiments.log",
) -> Path:
    """Записать результаты эксперимента в текстовый лог и вывести строку на экран."""
    if params is None:
        params = {}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / logfile

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {name} score={score:.6f} std={std:.6f} params={params}\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)

    print(line.strip())
    return path


def load_experiments(
    logfile: str = "experiments.log",
) -> pd.DataFrame:
    """Загрузить лог экспериментов как DataFrame (если файла нет — пустой DF)."""
    path = LOG_DIR / logfile
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "name", "score", "std", "params"])

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            timestamp = " ".join(parts[0:2])
            name = parts[2]
            score = float(parts[3].split("=")[1])
            std = float(parts[4].split("=")[1])
            params_str = " ".join(parts[5:]).replace("params=", "", 1)
            rows.append(
                {"timestamp": timestamp, "name": name, "score": score, "std": std, "params": params_str}
            )

    return pd.DataFrame(rows)