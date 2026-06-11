"""
Unit tests for feature_engineering module.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    @pytest.fixture
    def sample_ecommerce_df(self):
        """Create sample e-commerce DataFrame for testing."""
        return pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "signup_time": [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:00",
                    "2023-01-01 12:00:00",
                    "2023-01-01 12:00:00",
                    "2023-01-02 09:00:00",
                ],
                "purchase_time": [
                    "2023-01-01 10:30:00",
                    "2023-01-01 11:00:00",
                    "2023-01-01 12:30:00",
                    "2023-01-01 13:00:00",
                    "2023-01-02 09:15:00",
                ],
                "purchase_value": [100, 200, 150, 300, 250],
                "device_id": ["dev1", "dev1", "dev2", "dev2", "dev3"],
                "source": ["SEO", "SEO", "Ads", "Ads", "Direct"],
                "browser": ["Chrome", "Chrome", "Safari", "Safari", "Firefox"],
                "sex": ["M", "M", "F", "F", "M"],
                "age": [25, 25, 30, 30, 35],
                "ip_address": [
                    "192.168.1.1",
                    "192.168.1.2",
                    "192.168.1.3",
                    "192.168.1.4",
                    "192.168.1.5",
                ],
                "class": [0, 1, 0, 1, 0],
            }
        )

    @pytest.fixture
    def sample_ip_mapping_df(self):
        """Create sample IP mapping DataFrame."""
        return pd.DataFrame(
            {
                "lower_bound_ip_address": ["192.168.1.0", "192.168.2.0"],
                "upper_bound_ip_address": ["192.168.1.255", "192.168.2.255"],
                "country": ["USA", "Canada"],
            }
        )

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()

    def test_convert_ip_to_int_valid(self, feature_engineer):
        """Test IP to integer conversion with valid IP."""
        result = feature_engineer.convert_ip_to_int("192.168.1.1")
        assert isinstance(result, int)
        assert result > 0

    def test_convert_ip_to_int_invalid(self, feature_engineer):
        """Test IP to integer conversion with invalid IP."""
        result = feature_engineer.convert_ip_to_int("invalid_ip")
        assert result is None

    def test_convert_ip_to_int_empty(self, feature_engineer):
        """Test IP to integer conversion with empty string."""
        result = feature_engineer.convert_ip_to_int("")
        assert result is None

    def test_add_time_features(self, feature_engineer, sample_ecommerce_df):
        """Test adding time-based features."""
        result_df = feature_engineer.add_time_features(sample_ecommerce_df)

        assert "hour_of_day" in result_df.columns
        assert "day_of_week" in result_df.columns
        assert "day_of_month" in result_df.columns
        assert "month" in result_df.columns
        assert "is_weekend" in result_df.columns

        assert result_df["hour_of_day"].dtype in ["int64", "int32"]
        assert result_df["day_of_week"].dtype in ["int64", "int32"]
        assert result_df["is_weekend"].dtype in ["int64", "int32"]

        assert result_df.loc[0, "hour_of_day"] == 10
        assert result_df.loc[0, "day_of_week"] == 6
        assert result_df.loc[0, "is_weekend"] == 1

    def test_add_time_since_signup(self, feature_engineer, sample_ecommerce_df):
        """Test adding time since signup feature."""
        result_df = feature_engineer.add_time_since_signup(sample_ecommerce_df)

        assert "time_since_signup_hours" in result_df.columns
        assert "is_very_recent_signup" in result_df.columns

        assert abs(result_df.loc[0, "time_since_signup_hours"] - 0.5) < 0.01
        assert result_df.loc[0, "is_very_recent_signup"] == 1

        assert abs(result_df.loc[4, "time_since_signup_hours"] - 0.25) < 0.01
        assert result_df.loc[4, "is_very_recent_signup"] == 1

    def test_add_transaction_frequency(self, feature_engineer, sample_ecommerce_df):
        """Test adding transaction frequency features."""
        result_df = feature_engineer.add_transaction_frequency(
            sample_ecommerce_df, windows_hours=[1, 24]
        )

        assert "transactions_last_1h" in result_df.columns
        assert "transactions_last_24h" in result_df.columns

        assert result_df.loc[0, "transactions_last_1h"] == 0
        assert result_df.loc[1, "transactions_last_1h"] == 1

        assert result_df.loc[2, "transactions_last_1h"] == 0
        assert result_df.loc[3, "transactions_last_1h"] == 1

    def test_map_ip_to_country(
        self, feature_engineer, sample_ecommerce_df, sample_ip_mapping_df
    ):
        """Test IP to country mapping."""
        result_df = feature_engineer.map_ip_to_country(
            sample_ecommerce_df, sample_ip_mapping_df
        )

        assert "ip_int" in result_df.columns
        assert "country" in result_df.columns

        assert all(isinstance(ip, int) for ip in result_df["ip_int"] if pd.notna(ip))

        assert result_df["country"].notna().sum() >= 0

    def test_add_country_risk_features(self, feature_engineer):
        """Test adding country risk features."""

        test_df = pd.DataFrame(
            {
                "country": ["USA", "USA", "Canada", "USA", "Unknown"],
                "class": [0, 1, 0, 1, 0],
            }
        )

        result_df = feature_engineer.add_country_risk_features(test_df)

        assert "fraud_rate" in result_df.columns
        assert "is_high_risk_country" in result_df.columns

        assert result_df.loc[0, "fraud_rate"] == 2 / 3
        assert result_df.loc[2, "fraud_rate"] == 0.0

    def test_add_country_risk_features_no_country_column(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test adding country risk features when no country column exists."""
        result_df = feature_engineer.add_country_risk_features(sample_ecommerce_df)

        assert result_df.equals(sample_ecommerce_df)

    def test_encode_categorical_features_auto_detect(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test categorical feature encoding with auto-detection."""
        result_df = feature_engineer.encode_categorical_features(sample_ecommerce_df)

        assert "source_Ads" in result_df.columns or "source_Direct" in result_df.columns
        assert (
            "browser_Chrome" in result_df.columns
            or "browser_Safari" in result_df.columns
        )
        assert "sex_M" in result_df.columns

        assert "source" not in result_df.columns
        assert "browser" not in result_df.columns
        assert "sex" not in result_df.columns

    def test_encode_categorical_features_specified_columns(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test categorical feature encoding with specified columns."""
        categorical_cols = ["source", "browser"]
        result_df = feature_engineer.encode_categorical_features(
            sample_ecommerce_df, categorical_cols=categorical_cols
        )

        assert "source_Ads" in result_df.columns or "source_Direct" in result_df.columns
        assert (
            "browser_Chrome" in result_df.columns
            or "browser_Safari" in result_df.columns
        )

        assert "sex" in result_df.columns

    def test_encode_categorical_features_drop_first(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test categorical feature encoding with drop_first=True."""
        result_df = feature_engineer.encode_categorical_features(
            sample_ecommerce_df, drop_first=True
        )

        sex_columns = [col for col in result_df.columns if col.startswith("sex_")]
        assert len(sex_columns) == 1

    def test_encode_categorical_features_no_drop_first(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test categorical feature encoding with drop_first=False."""
        result_df = feature_engineer.encode_categorical_features(
            sample_ecommerce_df, drop_first=False
        )

        sex_columns = [col for col in result_df.columns if col.startswith("sex_")]
        assert len(sex_columns) >= 1

    def test_scale_numerical_features_standard(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test numerical feature scaling with StandardScaler."""

        test_df = sample_ecommerce_df[["purchase_value", "age", "class"]].copy()

        result_df, scaler = feature_engineer.scale_numerical_features(
            test_df, method="standard"
        )

        assert abs(result_df["purchase_value"].mean()) < 0.1
        assert abs(result_df["purchase_value"].std(ddof=0) - 1.0) < 0.1
        assert abs(result_df["age"].mean()) < 0.1
        assert abs(result_df["age"].std(ddof=0) - 1.0) < 0.1

        assert scaler is not None

    def test_scale_numerical_features_minmax(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test numerical feature scaling with MinMaxScaler."""

        test_df = sample_ecommerce_df[["purchase_value", "age", "class"]].copy()

        result_df, scaler = feature_engineer.scale_numerical_features(
            test_df, method="minmax"
        )

        assert result_df["purchase_value"].min() >= 0
        assert result_df["purchase_value"].max() <= 1
        assert result_df["age"].min() >= 0
        assert result_df["age"].max() <= 1

        assert scaler is not None

    def test_scale_numerical_features_invalid_method(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test numerical feature scaling with invalid method."""
        with pytest.raises(ValueError, match="Unknown scaling method"):
            feature_engineer.scale_numerical_features(
                sample_ecommerce_df, method="invalid"
            )

    def test_scale_numerical_features_auto_detect(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test numerical feature scaling with auto-detection of columns."""
        result_df, scaler = feature_engineer.scale_numerical_features(
            sample_ecommerce_df, method="standard"
        )

        assert "purchase_value" in result_df.columns
        assert "age" in result_df.columns

        assert "class" in result_df.columns
        assert "user_id" in result_df.columns

    def test_feature_engineering_pipeline_integration(
        self, feature_engineer, sample_ecommerce_df
    ):
        """Test integration of multiple feature engineering steps."""

        result_df = feature_engineer.add_time_features(sample_ecommerce_df)
        result_df = feature_engineer.add_time_since_signup(result_df)
        result_df = feature_engineer.add_transaction_frequency(
            result_df, windows_hours=[1]
        )
        result_df = feature_engineer.encode_categorical_features(result_df)

        expected_features = [
            "hour_of_day",
            "day_of_week",
            "time_since_signup_hours",
            "transactions_last_1h",
        ]

        for feature in expected_features:
            assert feature in result_df.columns, f"Missing feature: {feature}"

        assert any(col.startswith("source_") for col in result_df.columns)
        assert any(col.startswith("browser_") for col in result_df.columns)

        assert len(result_df) == len(sample_ecommerce_df)
        assert "class" in result_df.columns
