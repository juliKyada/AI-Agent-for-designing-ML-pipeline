"""
Dataset metadata extraction utilities for MetaFlow
"""
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()
config = get_config()


class MetadataExtractor:
    """Extracts structural/statistical metadata from dataset features and target."""

    def __init__(self):
        self.metadata: Dict = {}

    def extract(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Extract metadata used by downstream components.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Metadata dictionary.
        """
        if X is None or y is None:
            raise ValueError("Both X and y are required for metadata extraction")
        if X.empty:
            raise ValueError("X is empty")

        categorical_threshold = config.get('data.categorical_threshold', 10)

        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [
            col for col in X.columns
            if col not in numerical_features
            or (X[col].nunique(dropna=True) <= categorical_threshold and X[col].dtype == 'object')
        ]

        categorical_features = list(dict.fromkeys(categorical_features))
        numerical_features = [col for col in numerical_features if col not in categorical_features]

        dataset_info = {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'target_name': y.name if y.name is not None else 'target',
        }

        features_info = {
            'all': X.columns.tolist(),
            'numerical': numerical_features,
            'categorical': categorical_features,
            'n_numerical': len(numerical_features),
            'n_categorical': len(categorical_features),
            'dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
        }

        target_info = {
            'name': y.name if y.name is not None else 'target',
            'dtype': str(y.dtype),
            'n_unique': int(y.nunique(dropna=True)),
            'missing': int(y.isna().sum()),
        }

        quality_info = {
            'missing_by_feature': X.isna().sum().to_dict(),
            'missing_ratio_by_feature': X.isna().mean().to_dict(),
            'total_missing_features': int(X.isna().sum().sum()),
            'duplicate_rows': int(X.duplicated().sum()),
        }

        statistics_info = {
            'numeric_summary': X[numerical_features].describe().to_dict() if numerical_features else {},
            'skewness': X[numerical_features].skew().to_dict() if numerical_features else {},
            'kurtosis': X[numerical_features].kurtosis().to_dict() if numerical_features else {},
            'target_skew': float(y.skew()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 0 else None,
            'target_kurtosis': float(y.kurtosis()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 0 else None,
            'target_summary': {
                'min': float(y.min()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 0 else None,
                'max': float(y.max()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 0 else None,
                'mean': float(y.mean()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 0 else None,
                'std': float(y.std()) if pd.api.types.is_numeric_dtype(y) and len(y.dropna()) > 1 else None,
            }
        }

        # Outlier detection (IQR based)
        outlier_info = {}
        if numerical_features:
            q1 = X[numerical_features].quantile(0.25)
            q3 = X[numerical_features].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_counts = ((X[numerical_features] < lower_bound) | (X[numerical_features] > upper_bound)).sum()
            outlier_info = {
                'counts': outlier_counts.to_dict(),
                'ratios': (outlier_counts / len(X)).to_dict(),
                'total_outliers': int(outlier_counts.sum()),
                'overall_density': float(outlier_counts.sum() / (len(X) * len(numerical_features))) if len(numerical_features) > 0 else 0
            }

        # Cardinality and type richness
        cardinality_info = {
            'categorical_cardinality': X[categorical_features].nunique().to_dict() if categorical_features else {},
            'high_cardinality_features': [col for col in categorical_features if X[col].nunique() > categorical_threshold],
            'constant_features': [col for col in X.columns if X[col].nunique() <= 1],
            'near_constant_features': [col for col in X.columns if X[col].value_counts(normalize=True).iloc[0] > 0.99] if not X.empty else [],
        }

        # Target correlation / Mutual Information (Simple correlation for now)
        correlation_info = {}
        if numerical_features and pd.api.types.is_numeric_dtype(y):
            correlations = X[numerical_features].corrwith(y).abs()
            correlation_info = {
                'feature_target_correlation': correlations.to_dict(),
                'mean_correlation': float(correlations.mean()),
                'max_correlation': float(correlations.max()),
                'top_correlated_features': correlations.sort_values(ascending=False).head(5).index.tolist()
            }

        # Class imbalance for classification
        imbalance_info = {}
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 20: # Heuristic for classification-like targets
            counts = y.value_counts()
            ratios = y.value_counts(normalize=True)
            imbalance_info = {
                'class_counts': counts.to_dict(),
                'class_ratios': ratios.to_dict(),
                'min_class_ratio': float(ratios.min()),
                'is_highly_imbalanced': float(ratios.min()) < 0.05
            }

        self.metadata = {
            'dataset': dataset_info,
            'features': features_info,
            'target': target_info,
            'quality': quality_info,
            'statistics': statistics_info,
            'outliers': outlier_info,
            'cardinality': cardinality_info,
            'correlation': correlation_info,
            'imbalance': imbalance_info
        }

        logger.info(
            f"Metadata extracted: {dataset_info['n_samples']} samples, "
            f"{dataset_info['n_features']} features "
            f"({features_info['n_numerical']} numerical, {features_info['n_categorical']} categorical)"
        )

        return self.metadata

    def get_summary(self) -> str:
        """Return a human-readable metadata summary."""
        if not self.metadata:
            return "No metadata available. Run extract(X, y) first."

        ds = self.metadata['dataset']
        features = self.metadata['features']
        target = self.metadata['target']
        quality = self.metadata['quality']

        lines = [
            f"Dataset: {ds['n_samples']} samples, {ds['n_features']} features",
            f"Features: {features['n_numerical']} numerical, {features['n_categorical']} categorical",
            f"Target: {target['name']} (dtype={target['dtype']}, unique={target['n_unique']})",
            f"Missing feature values: {quality['total_missing_features']}",
            f"Duplicate feature rows: {quality['duplicate_rows']}",
        ]

        return "\n".join(lines)
