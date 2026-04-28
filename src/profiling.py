import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

class DataProfiler:
    """
    Универсальный профилировщик табличных данных.
    Не зависит от конкретного датасета.
    """

    # --- Порог кардинальности: если уникальных > X% от длины, считаем колонку "high-cardinality" ---
    CATEGORICAL_NUNIQUE_THRESHOLD = 0.05  # 5% от размера датасета

    def __init__(self, df: pd.DataFrame, target: str = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()
        self.target = target

        # Кэши типов — рассчитываем один раз
        self._numeric_cols = None
        self._cat_cols = None
        self._datetime_cols = None
        self._bool_cols = None

        self._validate_target()

    def _validate_target(self):
        if self.target is not None and self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame columns: {list(self.df.columns)}")

    # =========================================================================
    # БАЗОВЫЕ МЕТОДЫ ОПРЕДЕЛЕНИЯ ТИПОВ КОЛОНОК
    # =========================================================================

    @property
    def numeric_cols(self) -> list:
        """Числовые колонки (int, float), исключая target."""
        if self._numeric_cols is None:
            cols = self.df.select_dtypes(include=['number']).columns.tolist()
            self._numeric_cols = [c for c in cols if c != self.target]
        return self._numeric_cols

    @property
    def cat_cols(self) -> list:
        """Категориальные колонки (object, category), исключая target."""
        if self._cat_cols is None:
            cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            self._cat_cols = [c for c in cols if c != self.target]
        return self._cat_cols

    @property
    def datetime_cols(self) -> list:
        """Datetime колонки."""
        if self._datetime_cols is None:
            self._datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        return self._datetime_cols

    @property
    def bool_cols(self) -> list:
        """Boolean колонки."""
        if self._bool_cols is None:
            self._bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        return self._bool_cols

    @property
    def all_feature_cols(self) -> list:
        """Все колонки кроме target."""
        return [c for c in self.df.columns if c != self.target]

    @staticmethod
    def _is_text_like(series) -> bool:
        return (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_categorical_dtype(series)
        )

    def get_col_type(self, col: str) -> str:
        """
        Определяет логический тип колонки.

        Returns:
            'numeric'      — числовой (int/float)
            'categorical'  — объект / category / бинарный int / низкая кардинальность
            'datetime'     — дата/время
            'boolean'      — булев тип
            'high_cardinality' — строковый с большим числом уникальных значений
            'unknown'      — не удалось определить
        """

        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        series = self.df[col]
        dtype = series.dtype

        if is_bool_dtype(dtype):
            return 'boolean'

        if is_datetime64_any_dtype(dtype):
            return 'datetime'

        if is_numeric_dtype(dtype):
            # Числовая с низкой кардинальностью (< 10 уник.) — скорее всего кодированная категория
            n_unique = series.nunique()
            if n_unique <= 10 and series.isin([0, 1]).all():
                return 'categorical'   # бинарная/флаговая
            return 'numeric'

        if self._is_text_like(dtype):
            n_unique = series.nunique()
            threshold = max(20, int(len(series) * self.CATEGORICAL_NUNIQUE_THRESHOLD))
            if n_unique > threshold:
                return 'high_cardinality'
            return 'categorical'

        return 'unknown'

    def get_col_pair_type(self, col1: str, col2: str) -> tuple:
        """
        Возвращает пару типов (type1, type2) для бивариантного анализа.
        Упрощает до ('numeric', 'categorical') или ('numeric', 'numeric') и т.д.
        """
        def _simplify(t):
            if t in ('numeric',):
                return 'numeric'
            if t in ('categorical', 'boolean', 'high_cardinality'):
                return 'categorical'
            return t

        return (_simplify(self.get_col_type(col1)),
                _simplify(self.get_col_type(col2)))

    # =========================================================================
    # УТИЛИТЫ
    # =========================================================================

    def _check_col(self, col: str):
        """Проверяет существование колонки."""
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(self.df.columns)}")

    def _check_target(self):
        """Проверяет, что target задан."""
        if self.target is None:
            raise ValueError("target is not set. Pass target='col_name' to DataProfiler()")

    def __repr__(self):
        return (
            f"DataProfiler(rows={len(self.df)}, cols={len(self.df.columns)}, "
            f"target='{self.target}', "
            f"numeric={len(self.numeric_cols)}, cat={len(self.cat_cols)})"
        )

    def overview(self) -> pd.DataFrame:
        """
        Сводная таблица по всем колонкам:
        тип dtype, логический тип, кол-во пропусков, % пропусков,
        кол-во уникальных значений, % уникальных.
        """
        rows = []
        for col in self.df.columns:
            s = self.df[col]
            n = len(s)
            n_null = int(s.isnull().sum())
            n_unique = int(s.nunique())
            rows.append({
                'column': col,
                'dtype': str(s.dtype),
                'logical_type': self.get_col_type(col),
                'is_target': col == self.target,
                'n_missing': n_null,
                'pct_missing': round(n_null / n * 100, 2),
                'n_unique': n_unique,
                'pct_unique': round(n_unique / n * 100, 2),
            })
        return pd.DataFrame(rows).set_index('column')

    def info_summary(self) -> dict:
        """
        Краткий словарь с ключевыми параметрами датасета:
        размерность, типы, пропуски, дубликаты, таргет, размер в памяти.
        """
        n_rows, n_cols = self.df.shape
        total_cells = n_rows * n_cols
        total_missing = int(self.df.isnull().sum().sum())
        type_counts = {}
        for col in self.df.columns:
            t = self.get_col_type(col)
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            'rows': n_rows,
            'columns': n_cols,
            'total_cells': total_cells,
            'total_missing': total_missing,
            'pct_missing': round(total_missing / total_cells * 100, 2),
            'duplicated_rows': int(self.df.duplicated().sum()),
            'target': self.target,
            'memory_mb': round(self.df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
            'col_types': type_counts,
        }

    def check_duplicates(self, subset: list = None, keep: str = 'first') -> pd.DataFrame:
        """
        Возвращает DataFrame дублирующихся строк.

        Параметры:
            subset  — список колонок для проверки (None = все колонки)
            keep    — какую копию считать оригиналом: 'first', 'last', False (отметить все)
        """
        mask = self.df.duplicated(subset=subset, keep=keep)
        n_dup = int(mask.sum())
        pct = round(n_dup / len(self.df) * 100, 2)

        print(f"Дублирующихся строк: {n_dup} ({pct}% от {len(self.df)})")
        if n_dup == 0:
            print("Дубликатов не найдено ✓")
            return pd.DataFrame()

        print(f"Колонки для проверки: {subset if subset else 'все'}")
        return self.df[mask].copy()

    def missing_summary(self) -> pd.DataFrame:
        """
        Сводка пропусков по колонкам: количество и процент.
        Колонки без пропусков по умолчанию не выводятся.
        """
        n_rows = len(self.df)
        data = []
        for col in self.df.columns:
            n_missing = int(self.df[col].isnull().sum())
            if n_missing == 0:
                continue  # можно сделать параметром, но по умолчанию скрываем
            data.append({
                'column': col,
                'n_missing': n_missing,
                'pct_missing': round(n_missing / n_rows * 100, 2),
            })
        if not data:
            return pd.DataFrame(columns=['n_missing', 'pct_missing'])
        df_miss = (
            pd.DataFrame(data)
            .set_index('column')
            .sort_values('pct_missing', ascending=False)
        )
        return df_miss

    def plot_missing(self, figsize=(8, 4), top_n=None):
        """
        Горизонтальный bar-chart пропусков по колонкам.

        Параметры:
            figsize : размер фигуры (width, height)
            top_n   : отобразить только top-N колонок по % пропусков
        """
        df_miss = self.missing_summary()
        if df_miss.empty:
            print("Пропусков нет ✓")
            return

        if top_n is not None:
            df_miss = df_miss.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(
            x='pct_missing',
            y=df_miss.index,
            data=df_miss.reset_index(),
            orient='h'
        )
        plt.xlabel('% пропусков')
        plt.ylabel('Колонка')
        plt.title('Доля пропусков по колонкам')
        plt.tight_layout()
        plt.show()

    def describe_numeric(self, include_target: bool = True) -> pd.DataFrame:
        """
        Расширенный аналог df.describe() по числовым колонкам:
        count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis, пропуски.
        """
        # Берём только числовые фичи
        num_cols = self.numeric_cols.copy()

        # При желании включаем target, если он числовой
        if include_target and self.target is not None and self.get_col_type(self.target) == 'numeric':
            if self.target not in num_cols:
                num_cols.append(self.target)

        if not num_cols:
            return pd.DataFrame()

        # Базовое describe()
        desc = self.df[num_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max

        # Добавляем пропуски и их долю
        desc["missing"] = self.df[num_cols].isnull().sum()
        desc["pct_missing"] = (desc["missing"] / len(self.df) * 100).round(2)

        # Добавляем skew и kurtosis
        desc["skew"] = self.df[num_cols].skew()
        desc["kurtosis"] = self.df[num_cols].kurtosis()

        # Порядок колонок
        cols_order = [
            "count", "mean", "std", "min", "25%", "50%", "75%", "max",
            "skew", "kurtosis", "missing", "pct_missing",
        ]
        return desc[cols_order]

    def describe_categorical(self, include_target: bool = True) -> pd.DataFrame:
        """
        Сводка по категориальным колонкам:
        n_unique, самое частое значение, его частота и доля, пропуски.
        """
        cat_cols = self.cat_cols.copy()

        # При желании включаем target, если он категориальный/булевый
        if include_target and self.target is not None and self.get_col_type(self.target) in ("categorical", "boolean"):
            if self.target not in cat_cols:
                cat_cols.append(self.target)

        if not cat_cols:
            return pd.DataFrame()

        rows = []
        n_rows = len(self.df)

        for col in cat_cols:
            s = self.df[col]
            n_missing = int(s.isnull().sum())
            value_counts = s.value_counts(dropna=True)

            if not value_counts.empty:
                top_value = value_counts.index[0]
                top_freq = int(value_counts.iloc[0])
                top_pct = round(top_freq / (n_rows - n_missing) * 100, 2)
            else:
                top_value = None
                top_freq = 0
                top_pct = 0.0

            rows.append(
                {
                    "column": col,
                    "n_unique": int(s.nunique(dropna=True)),
                    "top": top_value,
                    "top_freq": top_freq,
                    "top_pct": top_pct,
                    "missing": n_missing,
                    "pct_missing": round(n_missing / n_rows * 100, 2),
                }
            )

        df_cat = pd.DataFrame(rows).set_index("column")
        # Сортируем по кардинальности (по убыванию)
        return df_cat.sort_values("n_unique", ascending=False)

    def plot_distribution(self, col: str, bins: int = 30, kde: bool = True, figsize=(6, 4)):
        """
        Унивариантное распределение колонки:
          - числовая → гистограмма + KDE
          - категориальная/булева → countplot (bar)
        """
        self._check_col(col)
        col_type = self.get_col_type(col)
        s = self.df[col].dropna()

        plt.figure(figsize=figsize)

        if col_type == "numeric":
            sns.histplot(s, bins=bins, kde=kde)
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"Distribution of {col}")
        elif col_type in ("categorical", "boolean"):
            order = s.value_counts().index
            sns.countplot(x=s, order=order)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"Value counts of {col}")
        else:
            plt.close()
            raise ValueError(
                f"plot_distribution: unsupported column type '{col_type}' for column '{col}'"
            )

        plt.tight_layout()
        plt.show()

    def plot_all_distributions(self, bins: int = 30, kde: bool = True,
                               max_cols: int = 4, figsize=(14, 8)):
        """
        Грид гистограмм для всех числовых колонок.
        """
        num_cols = self.numeric_cols
        if not num_cols:
            print("Нет числовых колонок для отображения")
            return

        n = len(num_cols)
        n_cols = min(max_cols, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)  # даже если 1 ось

        for ax, col in zip(axes, num_cols):
            s = self.df[col].dropna()
            sns.histplot(s, bins=bins, kde=kde, ax=ax)
            ax.set_title(col)

        # Лишние оси отключаем
        for ax in axes[len(num_cols):]:
            ax.axis("off")

        fig.suptitle("Distributions of numeric features", y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_value_counts(self, col: str, top_n: int = 20,
                          normalize: bool = False, figsize=(6, 4)):
        """
        Горизонтальный bar chart топ-N категорий:
          - normalize=True → доли
          - normalize=False → абсолютные частоты
        """
        self._check_col(col)
        col_type = self.get_col_type(col)
        if col_type not in ("categorical", "boolean", "high_cardinality"):
            raise ValueError(
                f"plot_value_counts: column '{col}' is not categorical-like (got '{col_type}')"
            )

        s = self.df[col]
        vc = s.value_counts(normalize=normalize).head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(x=vc.values, y=vc.index, orient="h")
        plt.xlabel("Proportion" if normalize else "Count")
        plt.ylabel(col)
        plt.title(f"Top-{top_n} value counts of {col}")
        plt.tight_layout()
        plt.show()

    def detect_outliers_iqr(self, col: str, factor: float = 1.5) -> pd.Series:
        """
        Возвращает булеву маску выбросов по правилу IQR.

        Параметры:
            col    : числовая колонка
            factor : коэффициент для IQR, обычно 1.5
        """
        self._check_col(col)

        if self.get_col_type(col) != "numeric":
            raise ValueError(f"detect_outliers_iqr: column '{col}' must be numeric")

        s = self.df[col].dropna()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        mask = mask.fillna(False)

        return mask

    def outliers_summary_iqr(self, col: str, factor: float = 1.5) -> dict:
        """
        Краткая сводка по выбросам в колонке.
        """
        mask = self.detect_outliers_iqr(col, factor=factor)
        n_outliers = int(mask.sum())
        pct_outliers = round(n_outliers / len(self.df) * 100, 2)

        s = self.df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        return {
            "column": col,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
        }

    def plot_boxplot(self, col: str, by: str = None, figsize=(6, 4)):
        """
        Boxplot:
          - без by → обычный boxplot
          - с by   → групповой boxplot
        """
        self._check_col(col)

        if self.get_col_type(col) != "numeric":
            raise ValueError(f"plot_boxplot: column '{col}' must be numeric")

        plt.figure(figsize=figsize)

        if by is None:
            sns.boxplot(y=self.df[col])
            plt.ylabel(col)
            plt.title(f"Boxplot of {col}")
        else:
            self._check_col(by)
            sns.boxplot(x=self.df[by], y=self.df[col])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(by)
            plt.ylabel(col)
            plt.title(f"Boxplot of {col} by {by}")

        plt.tight_layout()
        plt.show()

    def plot_all_boxplots(self, max_cols: int = 4, figsize=(14, 8)):
        """
        Грид boxplot-ов для всех числовых колонок.
        """
        num_cols = self.numeric_cols
        if not num_cols:
            print("Нет числовых колонок для отображения")
            return

        n = len(num_cols)
        n_cols = min(max_cols, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, num_cols):
            sns.boxplot(y=self.df[col], ax=ax)
            ax.set_title(col)

        for ax in axes[len(num_cols):]:
            ax.axis("off")

        fig.suptitle("Boxplots of numeric features", y=1.02)
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self, include_target: bool = True, method: str = "pearson") -> pd.DataFrame:
        """
        Матрица корреляций по числовым колонкам.

        Параметры:
            include_target : включать ли target, если он числовой
            method         : 'pearson', 'spearman', 'kendall'
        """
        num_cols = self.numeric_cols.copy()

        if include_target and self.target is not None and self.get_col_type(self.target) == "numeric":
            if self.target not in num_cols:
                num_cols.append(self.target)

        if not num_cols:
            return pd.DataFrame()

        return self.df[num_cols].corr(method=method)

    def plot_corr_heatmap(self, include_target: bool = True, method: str = "pearson",
                          annot: bool = True, cmap: str = "coolwarm", figsize=(8, 6)):
        """
        Тепловая карта корреляций по числовым колонкам.
        """
        corr = self.correlation_matrix(include_target=include_target, method=method)

        if corr.empty:
            print("Нет числовых колонок для корреляционного анализа")
            return

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f", square=True)
        plt.title(f"Correlation heatmap ({method})")
        plt.tight_layout()
        plt.show()

    def plot_target_correlation(self, method: str = "pearson", figsize=(6, 4)):
        """
        Горизонтальный barplot корреляции числовых признаков с target.
        Работает только если target числовой.
        """
        self._check_target()

        if self.get_col_type(self.target) != "numeric":
            raise ValueError("plot_target_correlation: target must be numeric")

        corr = self.correlation_matrix(include_target=True, method=method)

        if self.target not in corr.columns:
            print("Target не попал в матрицу корреляций")
            return

        target_corr = corr[self.target].drop(index=self.target).sort_values()

        plt.figure(figsize=figsize)
        sns.barplot(x=target_corr.values, y=target_corr.index, orient="h")
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.title(f"Correlation with target: {self.target}")
        plt.tight_layout()
        plt.show()

    def plot_feature_vs_target(self, col: str, figsize=(6, 4)):
        """
        Автоматически выбирает тип графика в зависимости от типов
        признака и target.

        Сценарии:
          numeric -> numeric       : regplot
          categorical -> numeric   : boxplot
          numeric -> categorical   : boxplot / violinplot
          categorical -> categorical : countplot(hue=target)
        """
        self._check_col(col)
        self._check_target()

        if col == self.target:
            raise ValueError("plot_feature_vs_target: col must be different from target")

        col_type, target_type = self.get_col_pair_type(col, self.target)

        plt.figure(figsize=figsize)

        # 1. numeric -> numeric
        if col_type == "numeric" and target_type == "numeric":
            sns.regplot(x=self.df[col], y=self.df[self.target], scatter_kws={"alpha": 0.5})
            plt.xlabel(col)
            plt.ylabel(self.target)
            plt.title(f"{col} vs {self.target}")

        # 2. categorical -> numeric
        elif col_type == "categorical" and target_type == "numeric":
            sns.boxplot(x=self.df[col], y=self.df[self.target])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(col)
            plt.ylabel(self.target)
            plt.title(f"{col} vs {self.target}")

        # 3. numeric -> categorical
        elif col_type == "numeric" and target_type == "categorical":
            sns.boxplot(x=self.df[self.target], y=self.df[col])
            plt.xlabel(self.target)
            plt.ylabel(col)
            plt.title(f"{col} by {self.target}")

        # 4. categorical -> categorical
        elif col_type == "categorical" and target_type == "categorical":
            sns.countplot(x=self.df[col], hue=self.df[self.target])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"{col} vs {self.target}")

        else:
            plt.close()
            raise ValueError(
                f"Unsupported pair type: ({col_type}, {target_type}) for ({col}, {self.target})"
            )

        plt.tight_layout()
        plt.show()

    def crosstab(self, col1: str, col2: str, normalize: str = "index") -> pd.DataFrame:
        """
        Возвращает кросс-таблицу для двух категориальных признаков.

        normalize:
          - False / None : абсолютные значения
          - 'index'      : нормировка по строкам
          - 'columns'    : нормировка по столбцам
          - 'all'        : нормировка по всей таблице
        """
        self._check_col(col1)
        self._check_col(col2)

        return pd.crosstab(self.df[col1], self.df[col2], normalize=normalize)

    def plot_crosstab_heatmap(self, col1: str, col2: str, normalize: str = "index",
                              cmap: str = "Blues", figsize=(6, 4)):
        """
        Тепловая карта кросс-таблицы для двух категориальных признаков.
        """
        ct = self.crosstab(col1, col2, normalize=normalize)

        plt.figure(figsize=figsize)
        sns.heatmap(ct, annot=True, cmap=cmap, fmt=".2f")
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f"Crosstab heatmap: {col1} vs {col2}")
        plt.tight_layout()
        plt.show()

    def check_multicollinearity(self, threshold: float = 0.9) -> pd.DataFrame:
        """
        Возвращает пары числовых признаков с высокой корреляцией по модулю.
        """
        corr = self.correlation_matrix(include_target=False)

        if corr.empty:
            return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

        rows = []
        cols = corr.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= threshold:
                    rows.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(val, 4),
                    })

        if not rows:
            return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

        return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    def target_summary(self) -> pd.DataFrame:
        """
        Краткая сводка по target.
        Для категориального target: counts + доли.
        Для числового target: describe().
        """
        self._check_target()

        target_type = self.get_col_type(self.target)

        if target_type == "numeric":
            return self.df[self.target].describe().to_frame(name=self.target)

        counts = self.df[self.target].value_counts(dropna=False)
        ratios = self.df[self.target].value_counts(dropna=False, normalize=True)

        result = pd.DataFrame({
            "count": counts,
            "ratio": ratios.round(4)
        })

        return result

    def plot_target_distribution(self, figsize=(6, 4)):
        """
        Распределение target:
          - categorical -> countplot
          - numeric     -> histplot
        """
        self._check_target()

        target_type = self.get_col_type(self.target)

        plt.figure(figsize=figsize)

        if target_type == "numeric":
            sns.histplot(self.df[self.target].dropna(), kde=True)
            plt.xlabel(self.target)
            plt.ylabel("Count")
            plt.title(f"Distribution of {self.target}")
        else:
            order = self.df[self.target].value_counts(dropna=False).index
            sns.countplot(x=self.df[self.target], order=order)
            plt.xlabel(self.target)
            plt.ylabel("Count")
            plt.title(f"Distribution of {self.target}")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    def cardinality_summary(self) -> pd.DataFrame:
        """
        Сводка по кардинальности категориальных признаков.
        """
        rows = []

        for col in self.cat_cols:
            rows.append({
                "column": col,
                "nunique": int(self.df[col].nunique(dropna=True)),
                "n_missing": int(self.df[col].isna().sum()),
                "top_value": self.df[col].mode(dropna=True).iloc[0] if not self.df[col].mode(
                    dropna=True).empty else None,
                "top_freq": int(self.df[col].value_counts(dropna=True).iloc[0]) if not self.df[
                    col].dropna().empty else 0,
            })

        if not rows:
            return pd.DataFrame(columns=["column", "nunique", "n_missing", "top_value", "top_freq"])

        return pd.DataFrame(rows).sort_values("nunique", ascending=False).reset_index(drop=True)

    def constant_columns(self) -> list:
        """
        Возвращает список константных колонок
        (у которых только одно уникальное значение без учета NaN).
        """
        cols = []

        for col in self.all_feature_cols:
            if self.df[col].nunique(dropna=True) <= 1:
                cols.append(col)

        return cols

    def low_variance_columns(self, threshold: float = 0.01) -> list:
        """
        Возвращает список числовых колонок с низкой дисперсией.
        threshold: порог по variance()
        """
        cols = []

        for col in self.numeric_cols:
            if self.df[col].dropna().empty:
                continue
            if self.df[col].var() <= threshold:
                cols.append(col)

        return cols

    def quasi_constant_columns(self, threshold: float = 0.99) -> list:
        """
        Возвращает колонки, где одно значение занимает >= threshold доли наблюдений.
        Подходит и для numeric, и для categorical.
        """
        cols = []

        for col in self.all_feature_cols:
            vc = self.df[col].value_counts(dropna=False, normalize=True)
            if not vc.empty and vc.iloc[0] >= threshold:
                cols.append(col)

        return cols

    def plot_pairplot(self, cols: list = None, hue: str = None, max_cols: int = 5):
        """
        Pairplot по числовым признакам.

        cols:
            список колонок; если None — берутся первые max_cols числовых колонок
        hue:
            колонка для цветового разбиения, например target
        """
        if cols is None:
            cols = self.numeric_cols[:max_cols]

        cols = [c for c in cols if c in self.df.columns]

        if len(cols) < 2:
            raise ValueError("plot_pairplot: need at least 2 columns")

        data = self.df[cols].copy()

        if hue is not None:
            self._check_col(hue)
            data[hue] = self.df[hue]

        sns.pairplot(data=data, hue=hue, dropna=True, diag_kind="hist")
        plt.show()

    def run_basic_eda(self, show_plots: bool = True):
        """
        Быстрый базовый EDA-сценарий.
        Возвращает словарь с основными таблицами.
        """
        report = {
            "overview": self.overview(),
            "duplicates": self.check_duplicates(),
            "missing": self.missing_summary(),
            "numeric_summary": self.describe_numeric(),
            "categorical_summary": self.describe_categorical(),
            "cardinality": self.cardinality_summary(),
            "constant_columns": self.constant_columns(),
            "quasi_constant_columns": self.quasi_constant_columns(),
        }

        if self.target is not None:
            report["target_summary"] = self.target_summary()

        if show_plots:
            self.plot_missing()

            if self.target is not None:
                self.plot_target_distribution()

        return report

    def full_report(self) -> dict:
        """
        Полная сводка по датасету в виде словаря.
        Без генерации файлов, только Python-объект.
        """
        report = {
            "overview": self.overview(),
            "duplicates": self.check_duplicates(),
            "missing_summary": self.missing_summary(),
            "numeric_summary": self.describe_numeric(),
            "categorical_summary": self.describe_categorical(),
            "cardinality_summary": self.cardinality_summary(),
            "constant_columns": self.constant_columns(),
            "quasi_constant_columns": self.quasi_constant_columns(),
            "low_variance_columns": self.low_variance_columns(),
        }

        if len(self.numeric_cols) >= 2:
            report["correlation_matrix"] = self.correlation_matrix(include_target=False)
            report["multicollinearity"] = self.check_multicollinearity()

        if self.target is not None:
            report["target_summary"] = self.target_summary()

            try:
                report["target_correlation"] = self.target_correlation_summary()
            except Exception:
                pass

        return report

    def target_correlation_summary(self) -> pd.DataFrame:
        """
        Таблица корреляций числовых признаков с target.
        Работает только если target можно трактовать как numeric.
        """
        self._check_target()

        usable_cols = [c for c in self.numeric_cols if c != self.target]

        if self.target not in self.df.columns:
            raise ValueError("Target column not found")

        temp_cols = usable_cols + [self.target]
        corr = self.df[temp_cols].corr(numeric_only=True)

        if self.target not in corr.columns:
            raise ValueError("target_correlation_summary: target is not numeric for corr()")

        result = (
            corr[self.target]
            .drop(labels=[self.target], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .to_frame(name="correlation_with_target")
        )

        return result

    def print_report(self):
        """
        Красиво печатает основные куски full_report().
        """
        report = self.full_report()

        for key, value in report.items():
            print(f"\n{'=' * 20} {key.upper()} {'=' * 20}")
            print(value)





