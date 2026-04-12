"""
Data preprocessing utilities for handling missing values
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


class DataPreprocessor:
    """Handles data preprocessing including missing value handling"""
    
    def __init__(self, missing_threshold: float = 0.5, imputation_strategy: str = 'auto'):
        """
        Initialize DataPreprocessor
        
        Args:
            missing_threshold: Features with missing ratio > this will be removed (default: 0.5)
            imputation_strategy: Strategy for imputing missing values
                - 'auto': Use mean for numeric, mode for categorical
                - 'mean': Use mean for all numeric features
                - 'median': Use median for numeric features
                - 'mode': Use most frequent value
                - 'drop': Drop rows with any missing values
        """
        self.missing_threshold = missing_threshold
        self.imputation_strategy = imputation_strategy
        self.imputation_values = {}  # Store imputation values for consistency
        self.dropped_features = []  # Track dropped features
        self.rows_removed_by_target_na = 0  # Track rows removed due to target NaN
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform data in one step
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        logger.info("Starting data preprocessing...")
        
        # Step 1: Remove features with high missing ratio
        X_cleaned = self._remove_high_missing_features(X)
        
        # Step 2: Handle missing values in remaining features
        X_processed = self._impute_missing_values(X_cleaned)
        
        # Step 3: Handle rows with critical missing values in X
        initial_rows = len(X_processed)
        X_processed = X_processed.dropna(how='all')  # Drop rows that are completely empty
        dropped_rows = initial_rows - len(X_processed)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with all missing values in X")
        
        # Step 4: Remove rows with missing values in target variable
        y_processed = y
        if y is not None:
            y_processed = y[X_processed.index].copy()
            missing_in_y = y_processed.isna().sum()
            if missing_in_y > 0:
                logger.warning(f"Found {missing_in_y} missing values in target variable - removing rows")
                valid_indices = ~y_processed.isna()
                X_processed = X_processed[valid_indices]
                y_processed = y_processed[valid_indices]
                self.rows_removed_by_target_na = missing_in_y
        
        logger.info("Data preprocessing complete")
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted imputation values
        
        Args:
            X: Features DataFrame
            
        Returns:
            Processed DataFrame
        """
        X_copy = X.copy()
        
        # Remove columns that were dropped during fit
        X_copy = X_copy.drop(columns=[col for col in self.dropped_features if col in X_copy.columns])
        
        # Apply same imputation values used during fit
        for col, impute_val in self.imputation_values.items():
            if col in X_copy.columns and X_copy[col].isna().any():
                X_copy[col].fillna(impute_val, inplace=True)
        
        return X_copy
    
    def _remove_high_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with missing ratio exceeding threshold"""
        X_copy = X.copy()
        missing_ratio = X_copy.isna().mean()
        high_missing_cols = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        
        if high_missing_cols:
            logger.warning(
                f"Removing {len(high_missing_cols)} feature(s) with missing ratio > {self.missing_threshold}: "
                f"{high_missing_cols}"
            )
            self.dropped_features.extend(high_missing_cols)
            X_copy = X_copy.drop(columns=high_missing_cols)
        
        return X_copy
    
    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values based on strategy and feature type
        
        Args:
            X: Features DataFrame with some missing values
            
        Returns:
            DataFrame with imputed values
        """
        X_copy = X.copy()
        missing_info = X_copy.isna().sum()
        
        if missing_info.sum() == 0:
            logger.info("No missing values detected")
            return X_copy
        
        cols_with_missing = missing_info[missing_info > 0].index.tolist()
        logger.info(f"Found missing values in {len(cols_with_missing)} feature(s)")
        
        for col in cols_with_missing:
            missing_count = X_copy[col].isna().sum()
            missing_pct = (missing_count / len(X_copy)) * 100
            
            # Determine if feature is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(X_copy[col])
            
            # Apply imputation strategy
            if self.imputation_strategy == 'drop':
                # Drop rows with missing values in this column
                X_copy = X_copy.dropna(subset=[col])
                logger.info(f"  {col}: Dropped {missing_count} rows ({missing_pct:.2f}% missing)")
                
            elif self.imputation_strategy == 'mode':
                # Use most frequent value
                impute_val = X_copy[col].mode()[0] if len(X_copy[col].mode()) > 0 else X_copy[col].value_counts().index[0]
                X_copy[col] = X_copy[col].fillna(impute_val)
                self.imputation_values[col] = impute_val
                logger.info(f"  {col}: Imputed with mode='{impute_val}' ({missing_pct:.2f}% missing)")
                
            elif self.imputation_strategy == 'median' and is_numeric:
                # Use median for numeric features
                impute_val = X_copy[col].median()
                X_copy[col] = X_copy[col].fillna(impute_val)
                self.imputation_values[col] = impute_val
                logger.info(f"  {col}: Imputed with median={impute_val:.4f} ({missing_pct:.2f}% missing)")
                
            elif self.imputation_strategy == 'mean' and is_numeric:
                # Use mean for numeric features
                impute_val = X_copy[col].mean()
                X_copy[col] = X_copy[col].fillna(impute_val)
                self.imputation_values[col] = impute_val
                logger.info(f"  {col}: Imputed with mean={impute_val:.4f} ({missing_pct:.2f}% missing)")
                
            else:  # Default: 'auto' strategy
                if is_numeric:
                    # Use median for numeric (more robust than mean)
                    impute_val = X_copy[col].median()
                    X_copy[col] = X_copy[col].fillna(impute_val)
                    self.imputation_values[col] = impute_val
                    logger.info(f"  {col}: Imputed with median={impute_val:.4f} ({missing_pct:.2f}% missing)")
                else:
                    # Use mode for categorical
                    mode_result = X_copy[col].mode()
                    impute_val = mode_result[0] if len(mode_result) > 0 else 'MISSING'
                    X_copy[col] = X_copy[col].fillna(impute_val)
                    self.imputation_values[col] = impute_val
                    logger.info(f"  {col}: Imputed with mode='{impute_val}' ({missing_pct:.2f}% missing)")
        
        return X_copy
    
    def get_preprocessing_report(self, X: pd.DataFrame) -> Dict:
        """
        Generate a preprocessing report
        
        Args:
            X: Original DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        missing_by_col = X.isna().sum()
        missing_ratio = X.isna().mean()
        
        report = {
            'total_missing_values': int(X.isna().sum().sum()),
            'features_with_missing': int((X.isna().sum() > 0).sum()),
            'total_features': X.shape[1],
            'removed_features': self.dropped_features,
            'rows_removed_by_target_na': self.rows_removed_by_target_na,
            'imputation_strategy': self.imputation_strategy,
            'imputation_values': self.imputation_values,
            'missing_by_feature': missing_by_col.to_dict(),
            'missing_ratio_by_feature': missing_ratio.to_dict(),
        }
        
        return report
