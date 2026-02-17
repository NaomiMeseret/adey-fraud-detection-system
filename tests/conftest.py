"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(level=logging.INFO)
    
    # Suppress noisy loggers during tests
    noisy_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'botocore',
        'boto3',
        's3fs',
        'fsspec'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(10),
        'value': np.random.randn(10),
        'category': np.random.choice(['A', 'B', 'C'], 10),
        'target': np.random.choice([0, 1], 10)
    })


@pytest.fixture
def temp_csv_file(sample_dataframe, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "sample.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent
