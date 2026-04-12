"""
Data loading utilities for MetaFlow
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()
config = get_config()


class DataLoader:
    """Handles loading datasets from files or DataFrames."""

    def __init__(self):
        self._last_df: Optional[pd.DataFrame] = None
        self._last_target_column: Optional[str] = None
        self._target_encoder = None  # sklearn LabelEncoder when target is categorical

    def load(self, file_path: Union[str, Path], target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a dataset from file.

        Args:
            file_path: Path to a CSV, Excel, or Parquet file.
            target_column: Name of target column. If not provided, uses the last column.

        Returns:
            Tuple (X, y) where X is features DataFrame and y is target Series.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(path)
        elif suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(path)
        elif suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .xlsx, .xls, .parquet")

        logger.info(f"Loaded dataset from {path}: {len(df)} rows, {len(df.columns)} columns")
        return self.load_from_dataframe(df, target_column)

    def load_from_dataframe(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and target from DataFrame.

        Args:
            df: Input DataFrame.
            target_column: Name of target column. If not provided, uses the last column.

        Returns:
            Tuple (X, y) where X is features DataFrame and y is target Series.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        working_df = df.copy()

        if target_column is None:
            target_column = working_df.columns[-1]

        if target_column not in working_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        y = working_df[target_column].copy()
        X = working_df.drop(columns=[target_column]).copy()

        # Drop rows with missing target so downstream models get valid labels
        valid = ~y.isna()
        if not valid.all():
            n_drop = (~valid).sum()
            logger.warning(f"Dropping {n_drop} rows with missing target")
            X = X.loc[valid].copy()
            y = y.loc[valid].copy()

        # Encode categorical target to 0,1,2,... so sklearn/XGBoost/LightGBM can train
        if not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            self._target_encoder = LabelEncoder()
            y_encoded = self._target_encoder.fit_transform(y.astype(str))
            y = pd.Series(y_encoded, index=y.index, name=y.name, dtype=int)
            logger.info(f"Target encoded to integers: {dict(zip(self._target_encoder.classes_, range(len(self._target_encoder.classes_))))}")
        else:
            self._target_encoder = None

        max_missing_ratio = config.get('data.max_missing_ratio', 0.5)
        if X.shape[1] > 0:
            missing_ratio = X.isna().mean()
            to_drop = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
            if to_drop:
                logger.warning(
                    f"Dropping {len(to_drop)} feature(s) with missing ratio > {max_missing_ratio}: {to_drop}"
                )
                X = X.drop(columns=to_drop)

        if X.shape[1] == 0:
            raise ValueError("No feature columns available after preprocessing")

        self._last_df = pd.concat([X, y.rename(target_column)], axis=1)
        self._last_target_column = target_column

        return X, y

    def get_basic_info(self) -> dict:
        """
        Get basic information for the most recently loaded dataset.

        Returns:
            Dictionary with row/column counts, target info, and missing-value summary.
        """
        if self._last_df is None:
            return {}

        df = self._last_df
        target_column = self._last_target_column

        return {
            'n_samples': int(len(df)),
            'n_columns': int(len(df.columns)),
            'target_column': target_column,
            'feature_columns': [c for c in df.columns if c != target_column],
            'missing_cells': int(df.isna().sum().sum()),
            'missing_ratio': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])) if df.shape[0] and df.shape[1] else 0.0,
        }
