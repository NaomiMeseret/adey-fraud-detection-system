"""
Feature engineering utilities for fraud detection project.
"""

import ipaddress
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for fraud detection datasets."""

    def __init__(self):
        """Initialize FeatureEngineer."""
        pass

    def convert_ip_to_int(self, ip_value: Any) -> Optional[int]:
        """
        Convert an IP address or numeric IP range boundary to integer.

        Args:
            ip_value: Dotted IPv4 address, integer-like string, int, or float

        Returns:
            Integer representation of IP address, or None if invalid
        """
        if pd.isna(ip_value):
            return None

        try:
            if isinstance(ip_value, (int, np.integer)):
                ip_int = int(ip_value)
            elif isinstance(ip_value, (float, np.floating)):
                if not np.isfinite(ip_value):
                    return None
                ip_int = int(ip_value)
            else:
                ip_str = str(ip_value).strip()
                try:
                    ip_int = int(float(ip_str))
                except ValueError:
                    ip_int = int(ipaddress.IPv4Address(ip_str))

            if 0 <= ip_int <= int(ipaddress.IPv4Address("255.255.255.255")):
                return ip_int
            return None
        except (ValueError, ipaddress.AddressValueError):
            return None

    def add_time_features(
        self, df: pd.DataFrame, time_col: str = "purchase_time"
    ) -> pd.DataFrame:
        """
        Add time-based features to the dataset.

        Args:
            df: Input DataFrame
            time_col: Name of the timestamp column

        Returns:
            DataFrame with added time features
        """
        df = df.copy()

        df[time_col] = pd.to_datetime(df[time_col])

        df["hour_of_day"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["day_of_month"] = df[time_col].dt.day
        df["month"] = df[time_col].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        logger.info(f"Added time features to DataFrame")
        return df

    def add_time_since_signup(
        self,
        df: pd.DataFrame,
        signup_col: str = "signup_time",
        purchase_col: str = "purchase_time",
    ) -> pd.DataFrame:
        """
        Add time since signup feature.

        Args:
            df: Input DataFrame
            signup_col: Name of signup time column
            purchase_col: Name of purchase time column

        Returns:
            DataFrame with time_since_signup feature
        """
        df = df.copy()

        df[signup_col] = pd.to_datetime(df[signup_col])
        df[purchase_col] = pd.to_datetime(df[purchase_col])

        df["time_since_signup_hours"] = (
            df[purchase_col] - df[signup_col]
        ).dt.total_seconds() / 3600

        df["is_very_recent_signup"] = (df["time_since_signup_hours"] < 1).astype(int)

        logger.info(f"Added time_since_signup features")
        return df

    def add_transaction_frequency(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        time_col: str = "purchase_time",
        windows_hours: List[int] = [1, 24, 168],
    ) -> pd.DataFrame:
        """
        Add transaction frequency features for different time windows.

        Args:
            df: Input DataFrame
            user_col: User identifier column
            time_col: Timestamp column
            windows_hours: List of time windows in hours

        Returns:
            DataFrame with frequency features
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        df = df.sort_values([user_col, time_col])

        for window in windows_hours:
            col_name = f"transactions_last_{window}h"
            df[col_name] = 0

            for user_id in df[user_col].unique():
                user_mask = df[user_col] == user_id
                user_times = df.loc[user_mask, time_col]

                for idx in df[user_mask].index:
                    current_time = df.loc[idx, time_col]
                    window_start = current_time - pd.Timedelta(hours=window)

                    count = (
                        (user_times >= window_start) & (user_times < current_time)
                    ).sum()
                    df.loc[idx, col_name] = count

        logger.info(
            f"Added transaction frequency features for windows: {windows_hours}"
        )
        return df

    def map_ip_to_country(
        self, df: pd.DataFrame, ip_mapping_df: pd.DataFrame, ip_col: str = "ip_address"
    ) -> pd.DataFrame:
        """
        Map IP addresses to countries using IP range mapping.

        Args:
            df: Transaction DataFrame
            ip_mapping_df: IP to country mapping DataFrame
            ip_col: IP address column name

        Returns:
            DataFrame with country column added
        """
        df = df.copy()

        df["ip_int"] = df[ip_col].apply(self.convert_ip_to_int)

        ip_mapping_df = ip_mapping_df.copy()
        ip_mapping_df["lower_bound_int"] = ip_mapping_df[
            "lower_bound_ip_address"
        ].apply(self.convert_ip_to_int)
        ip_mapping_df["upper_bound_int"] = ip_mapping_df[
            "upper_bound_ip_address"
        ].apply(self.convert_ip_to_int)

        ip_mapping_df = ip_mapping_df.dropna(
            subset=["lower_bound_int", "upper_bound_int"]
        )

        df["country"] = None

        for idx, row in df.iterrows():
            if pd.notna(row["ip_int"]):
                mask = (ip_mapping_df["lower_bound_int"] <= row["ip_int"]) & (
                    ip_mapping_df["upper_bound_int"] >= row["ip_int"]
                )
                matches = ip_mapping_df[mask]

                if len(matches) > 0:
                    df.loc[idx, "country"] = matches.iloc[0]["country"]

        logger.info(f"Mapped {df['country'].notna().sum()} IP addresses to countries")
        return df

    def add_country_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add country-level risk features based on fraud rates.

        Args:
            df: DataFrame with country column

        Returns:
            DataFrame with country risk features
        """
        df = df.copy()

        if "country" not in df.columns:
            logger.warning("Country column not found, skipping country risk features")
            return df

        country_stats = (
            df.groupby("country").agg({"class": ["count", "sum", "mean"]}).reset_index()
        )

        country_stats.columns = [
            "country",
            "total_transactions",
            "fraud_count",
            "fraud_rate",
        ]

        df = df.merge(
            country_stats[["country", "fraud_rate"]], on="country", how="left"
        )

        overall_fraud_rate = df["class"].mean()
        df["fraud_rate"] = df["fraud_rate"].fillna(overall_fraud_rate)

        fraud_rate_threshold = country_stats["fraud_rate"].quantile(0.9)
        df["is_high_risk_country"] = (df["fraud_rate"] >= fraud_rate_threshold).astype(
            int
        )

        logger.info("Added country risk features")
        return df

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        drop_first: bool = True,
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode
            drop_first: Whether to drop first category to avoid multicollinearity

        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()

        if categorical_cols is None:

            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        excluded_cols = {
            "class",
            "Class",
            "user_id",
            "device_id",
            "ip_address",
            "ip_int",
            "signup_time",
            "purchase_time",
        }
        categorical_cols = [col for col in categorical_cols if col not in excluded_cols]

        df_encoded = pd.get_dummies(
            df, columns=categorical_cols, drop_first=drop_first, prefix_sep="_"
        )

        logger.info(f"Encoded categorical columns: {categorical_cols}")
        logger.info(f"DataFrame shape after encoding: {df_encoded.shape}")

        return df_encoded

    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        method: str = "standard",
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns to scale
            method: Scaling method ('standard' or 'minmax')

        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        df = df.copy()

        if numerical_cols is None:

            exclude_cols = [
                "class",
                "Class",
                "user_id",
                "device_id",
                "ip_address",
                "ip_int",
            ]
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        logger.info(f"Scaled numerical columns using {method}: {numerical_cols}")

        return df, scaler
