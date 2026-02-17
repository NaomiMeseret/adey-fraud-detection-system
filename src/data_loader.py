"""
Data loading utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic validation of fraud detection datasets."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize DataLoader with data directory path."""
        if data_dir is None:
            # Default to project root/data/raw
            self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_ecommerce_data(self, filename: str = "Fraud_Data.csv") -> pd.DataFrame:
        """
        Load e-commerce fraud dataset.
        
        Args:
            filename: Name of the e-commerce dataset file
            
        Returns:
            DataFrame with e-commerce transaction data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"E-commerce data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = [
            'user_id', 'signup_time', 'purchase_time', 'purchase_value',
            'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in e-commerce data: {missing_columns}")
        
        logger.info(f"Loaded e-commerce data: {df.shape}")
        return df
    
    def load_creditcard_data(self, filename: str = "creditcard.csv") -> pd.DataFrame:
        """
        Load credit card fraud dataset.
        
        Args:
            filename: Name of the credit card dataset file
            
        Returns:
            DataFrame with credit card transaction data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Credit card data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['Time', 'V1', 'V28', 'Amount', 'Class']
        
        # Check for V1-V28 columns
        v_columns = [f'V{i}' for i in range(1, 29)]
        missing_v_cols = [col for col in v_columns if col not in df.columns]
        
        missing_columns = []
        if 'Time' not in df.columns:
            missing_columns.append('Time')
        if 'Amount' not in df.columns:
            missing_columns.append('Amount')
        if 'Class' not in df.columns:
            missing_columns.append('Class')
        missing_columns.extend(missing_v_cols)
        
        if missing_columns:
            raise ValueError(f"Missing required columns in credit card data: {missing_columns}")
        
        logger.info(f"Loaded credit card data: {df.shape}")
        return df
    
    def load_ip_country_mapping(self, filename: str = "IpAddress_to_Country.csv") -> pd.DataFrame:
        """
        Load IP address to country mapping dataset.
        
        Args:
            filename: Name of the IP mapping file
            
        Returns:
            DataFrame with IP to country mapping
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"IP country mapping file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in IP mapping data: {missing_columns}")
        
        logger.info(f"Loaded IP country mapping: {df.shape}")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_type: str) -> dict:
        """
        Perform basic data quality checks.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('ecommerce' or 'creditcard')
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        if dataset_type == 'ecommerce':
            # Check class distribution
            if 'class' in df.columns:
                class_dist = df['class'].value_counts().to_dict()
                results['class_distribution'] = class_dist
                results['imbalance_ratio'] = class_dist.get(0, 0) / max(class_dist.get(1, 1), 1)
        
        elif dataset_type == 'creditcard':
            # Check class distribution
            if 'Class' in df.columns:
                class_dist = df['Class'].value_counts().to_dict()
                results['class_distribution'] = class_dist
                results['imbalance_ratio'] = class_dist.get(0, 0) / max(class_dist.get(1, 1), 1)
        
        return results
