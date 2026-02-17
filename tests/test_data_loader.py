"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with test data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # Create test e-commerce data
            ecommerce_data = pd.DataFrame({
                'user_id': [1, 2, 3],
                'signup_time': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00'],
                'purchase_time': ['2023-01-01 10:30:00', '2023-01-02 11:30:00', '2023-01-03 12:30:00'],
                'purchase_value': [100, 200, 150],
                'device_id': ['dev1', 'dev2', 'dev3'],
                'source': ['SEO', 'Ads', 'Direct'],
                'browser': ['Chrome', 'Safari', 'Firefox'],
                'sex': ['M', 'F', 'M'],
                'age': [25, 30, 35],
                'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
                'class': [0, 1, 0]
            })
            ecommerce_data.to_csv(data_path / "Fraud_Data.csv", index=False)
            
            # Create test credit card data
            creditcard_data = pd.DataFrame({
                'Time': [0, 1, 2],
                'V1': [0.1, -0.2, 0.3],
                'V2': [0.2, -0.1, 0.1],
                'V3': [0.3, 0.1, -0.2],
                'V4': [0.1, 0.2, -0.1],
                'V5': [-0.1, 0.3, 0.2],
                'Amount': [100, 200, 150],
                'Class': [0, 1, 0]
            })
            # Add remaining V columns
            for i in range(6, 29):
                creditcard_data[f'V{i}'] = np.random.normal(0, 0.1, 3)
            
            creditcard_data.to_csv(data_path / "creditcard.csv", index=False)
            
            # Create test IP mapping data
            ip_mapping_data = pd.DataFrame({
                'lower_bound_ip_address': ['192.168.1.0', '192.168.2.0'],
                'upper_bound_ip_address': ['192.168.1.255', '192.168.2.255'],
                'country': ['USA', 'Canada']
            })
            ip_mapping_data.to_csv(data_path / "IpAddress_to_Country.csv", index=False)
            
            yield data_path
    
    @pytest.fixture
    def data_loader(self, temp_data_dir):
        """Create DataLoader instance with temporary data directory."""
        return DataLoader(temp_data_dir)
    
    def test_init_with_default_path(self):
        """Test DataLoader initialization with default path."""
        with patch('src.data_loader.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            loader = DataLoader()
            mock_path.assert_called_once()
    
    def test_init_with_custom_path(self, temp_data_dir):
        """Test DataLoader initialization with custom path."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_dir == temp_data_dir
    
    def test_init_with_nonexistent_path(self):
        """Test DataLoader initialization with nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            DataLoader(Path("/nonexistent/path"))
    
    def test_load_ecommerce_data_success(self, data_loader):
        """Test successful loading of e-commerce data."""
        df = data_loader.load_ecommerce_data()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 11)
        assert list(df.columns) == [
            'user_id', 'signup_time', 'purchase_time', 'purchase_value',
            'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
        ]
    
    def test_load_ecommerce_data_file_not_found(self, data_loader):
        """Test loading e-commerce data with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="E-commerce data file not found"):
            data_loader.load_ecommerce_data("nonexistent.csv")
    
    def test_load_ecommerce_data_missing_columns(self, temp_data_dir):
        """Test loading e-commerce data with missing required columns."""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'user_id': [1, 2],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-02 11:00:00'],
            # Missing other required columns
        })
        incomplete_data.to_csv(temp_data_dir / "incomplete.csv", index=False)
        
        loader = DataLoader(temp_data_dir)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_ecommerce_data("incomplete.csv")
    
    def test_load_creditcard_data_success(self, data_loader):
        """Test successful loading of credit card data."""
        df = data_loader.load_creditcard_data()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        assert 'Time' in df.columns
        assert 'Amount' in df.columns
        assert 'Class' in df.columns
        
        # Check for V1-V28 columns
        for i in range(1, 29):
            assert f'V{i}' in df.columns
    
    def test_load_creditcard_data_file_not_found(self, data_loader):
        """Test loading credit card data with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Credit card data file not found"):
            data_loader.load_creditcard_data("nonexistent.csv")
    
    def test_load_creditcard_data_missing_columns(self, temp_data_dir):
        """Test loading credit card data with missing required columns."""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'Time': [0, 1],
            'Amount': [100, 200]
            # Missing V columns and Class
        })
        incomplete_data.to_csv(temp_data_dir / "incomplete_cc.csv", index=False)
        
        loader = DataLoader(temp_data_dir)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_creditcard_data("incomplete_cc.csv")
    
    def test_load_ip_country_mapping_success(self, data_loader):
        """Test successful loading of IP country mapping."""
        df = data_loader.load_ip_country_mapping()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    
    def test_load_ip_country_mapping_file_not_found(self, data_loader):
        """Test loading IP mapping with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="IP country mapping file not found"):
            data_loader.load_ip_country_mapping("nonexistent.csv")
    
    def test_load_ip_country_mapping_missing_columns(self, temp_data_dir):
        """Test loading IP mapping with missing required columns."""
        incomplete_data = pd.DataFrame({
            'lower_bound_ip_address': ['192.168.1.0']
            # Missing other columns
        })
        incomplete_data.to_csv(temp_data_dir / "incomplete_ip.csv", index=False)
        
        loader = DataLoader(temp_data_dir)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_ip_country_mapping("incomplete_ip.csv")
    
    def test_validate_data_quality_ecommerce(self, data_loader):
        """Test data quality validation for e-commerce dataset."""
        df = data_loader.load_ecommerce_data()
        results = data_loader.validate_data_quality(df, 'ecommerce')
        
        assert 'shape' in results
        assert 'missing_values' in results
        assert 'duplicates' in results
        assert 'data_types' in results
        assert 'memory_usage_mb' in results
        assert 'class_distribution' in results
        assert 'imbalance_ratio' in results
        
        assert results['shape'] == (3, 11)
        assert results['class_distribution'] == {0: 2, 1: 1}
        assert results['imbalance_ratio'] == 2.0
    
    def test_validate_data_quality_creditcard(self, data_loader):
        """Test data quality validation for credit card dataset."""
        df = data_loader.load_creditcard_data()
        results = data_loader.validate_data_quality(df, 'creditcard')
        
        assert 'shape' in results
        assert 'class_distribution' in results
        assert 'imbalance_ratio' in results
        
        assert results['shape'][0] == 3
        assert results['class_distribution'] == {0: 2, 1: 1}
        assert results['imbalance_ratio'] == 2.0
    
    def test_validate_data_quality_with_duplicates(self, temp_data_dir):
        """Test data quality validation with duplicate rows."""
        # Create data with duplicates
        duplicate_data = pd.DataFrame({
            'user_id': [1, 1, 2],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-02 11:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 10:30:00', '2023-01-02 11:30:00'],
            'purchase_value': [100, 100, 200],
            'device_id': ['dev1', 'dev1', 'dev2'],
            'source': ['SEO', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Chrome', 'Safari'],
            'sex': ['M', 'M', 'F'],
            'age': [25, 25, 30],
            'ip_address': ['192.168.1.1', '192.168.1.1', '192.168.1.2'],
            'class': [0, 0, 1]
        })
        duplicate_data.to_csv(temp_data_dir / "duplicate_data.csv", index=False)
        
        loader = DataLoader(temp_data_dir)
        df = loader.load_ecommerce_data("duplicate_data.csv")
        results = loader.validate_data_quality(df, 'ecommerce')
        
        assert results['duplicates'] == 1  # One duplicate row
    
    def test_validate_data_quality_with_missing_values(self, temp_data_dir):
        """Test data quality validation with missing values."""
        # Create data with missing values
        missing_data = pd.DataFrame({
            'user_id': [1, 2, np.nan],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-02 11:30:00', '2023-01-03 12:30:00'],
            'purchase_value': [100, 200, 150],
            'device_id': ['dev1', 'dev2', 'dev3'],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Safari', 'Firefox'],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
            'class': [0, 1, 0]
        })
        missing_data.to_csv(temp_data_dir / "missing_data.csv", index=False)
        
        loader = DataLoader(temp_data_dir)
        df = loader.load_ecommerce_data("missing_data.csv")
        results = loader.validate_data_quality(df, 'ecommerce')
        
        assert results['missing_values']['user_id'] == 1
