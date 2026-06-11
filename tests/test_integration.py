"""
Integration tests for the fraud detection pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer


class TestIntegration:
    """Integration tests for the complete fraud detection pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with complete test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            np.random.seed(42)
            n_samples = 1000

            ecommerce_data = pd.DataFrame(
                {
                    "user_id": np.random.randint(1, 101, n_samples),
                    "signup_time": pd.date_range(
                        "2023-01-01", periods=n_samples, freq="h"
                    ),
                    "purchase_time": pd.date_range(
                        "2023-01-01", periods=n_samples, freq="h"
                    )
                    + pd.to_timedelta(np.random.randint(1, 72, n_samples), unit="h"),
                    "purchase_value": np.random.uniform(10, 500, n_samples),
                    "device_id": [
                        f"device_{i}" for i in np.random.randint(1, 51, n_samples)
                    ],
                    "source": np.random.choice(
                        ["SEO", "Ads", "Direct"], n_samples, p=[0.5, 0.3, 0.2]
                    ),
                    "browser": np.random.choice(
                        ["Chrome", "Safari", "Firefox"], n_samples, p=[0.6, 0.3, 0.1]
                    ),
                    "sex": np.random.choice(["M", "F"], n_samples, p=[0.6, 0.4]),
                    "age": np.random.randint(18, 65, n_samples),
                    "ip_address": [
                        f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                        for _ in range(n_samples)
                    ],
                    "class": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
                }
            )

            fraud_indices = np.random.choice(
                n_samples, size=int(n_samples * 0.05), replace=False
            )
            ecommerce_data.loc[fraud_indices, "purchase_value"] *= 1.5
            ecommerce_data.loc[fraud_indices, "source"] = "Ads"

            ecommerce_data.to_csv(data_path / "Fraud_Data.csv", index=False)

            creditcard_data = pd.DataFrame(
                {
                    "Time": np.arange(n_samples),
                    "Amount": np.random.uniform(1, 1000, n_samples),
                    "Class": np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
                }
            )

            for i in range(1, 29):
                creditcard_data[f"V{i}"] = np.random.normal(0, 0.5, n_samples)

            fraud_indices = creditcard_data[creditcard_data["Class"] == 1].index
            creditcard_data.loc[fraud_indices, "Amount"] *= 2.0

            creditcard_data.to_csv(data_path / "creditcard.csv", index=False)

            ip_ranges = []
            countries = [
                "USA",
                "Canada",
                "UK",
                "Germany",
                "France",
                "Japan",
                "Australia",
            ]

            for i, country in enumerate(countries):
                start_octet = i * 36
                end_octet = min(start_octet + 35, 255)
                ip_ranges.append(
                    {
                        "lower_bound_ip_address": f"192.168.{start_octet}.0",
                        "upper_bound_ip_address": f"192.168.{end_octet}.255",
                        "country": country,
                    }
                )

            ip_mapping_data = pd.DataFrame(ip_ranges)
            ip_mapping_data.to_csv(data_path / "IpAddress_to_Country.csv", index=False)

            yield data_path

    @pytest.fixture
    def pipeline_components(self, temp_data_dir):
        """Create all pipeline components."""
        data_loader = DataLoader(temp_data_dir)
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer(random_state=42)

        return data_loader, feature_engineer, model_trainer

    def test_complete_ecommerce_pipeline(self, pipeline_components):
        """Test complete e-commerce fraud detection pipeline."""
        data_loader, feature_engineer, model_trainer = pipeline_components

        ecommerce_df = data_loader.load_ecommerce_data()
        ip_mapping_df = data_loader.load_ip_country_mapping()

        assert ecommerce_df.shape[0] == 1000
        assert "class" in ecommerce_df.columns

        quality_results = data_loader.validate_data_quality(ecommerce_df, "ecommerce")
        assert "imbalance_ratio" in quality_results
        assert quality_results["imbalance_ratio"] > 1

        engineered_df = feature_engineer.add_time_features(ecommerce_df)
        engineered_df = feature_engineer.add_time_since_signup(engineered_df)

        engineered_df = feature_engineer.add_transaction_frequency(
            engineered_df, windows_hours=[1, 24]
        )

        engineered_df = feature_engineer.map_ip_to_country(engineered_df, ip_mapping_df)

        engineered_df = feature_engineer.add_country_risk_features(engineered_df)

        engineered_df = feature_engineer.encode_categorical_features(engineered_df)

        engineered_df, scaler = feature_engineer.scale_numerical_features(
            engineered_df, method="standard"
        )

        assert "hour_of_day" in engineered_df.columns
        assert "time_since_signup_hours" in engineered_df.columns
        assert "transactions_last_1h" in engineered_df.columns
        assert "fraud_rate" in engineered_df.columns
        assert "is_high_risk_country" in engineered_df.columns
        assert any(col.startswith("country_") for col in engineered_df.columns)

        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_df)

        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X_train, y_train, method="smote"
        )

        model_trainer.train_logistic_regression(X_resampled, y_resampled)
        model_trainer.train_random_forest(X_resampled, y_resampled)

        for model_name, model in model_trainer.models.items():
            results = model_trainer.evaluate_model(model, X_test, y_test, model_name)

            assert 0 <= results["accuracy"] <= 1
            assert 0 <= results["f1_score"] <= 1
            assert "roc_auc" in results

        comparison_df = model_trainer.compare_models(X_test, y_test)
        assert len(comparison_df) == 2
        assert "F1 Score" in comparison_df.columns

        best_model_name = comparison_df.iloc[0]["Model"]
        best_model = model_trainer.models[best_model_name]
        importance_df = model_trainer.get_feature_importance(
            best_model, X_test.columns.tolist(), best_model_name
        )

        assert not importance_df.empty
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

    def test_creditcard_pipeline(self, pipeline_components):
        """Test credit card fraud detection pipeline."""
        data_loader, feature_engineer, model_trainer = pipeline_components

        creditcard_df = data_loader.load_creditcard_data()

        assert creditcard_df.shape[0] == 1000
        assert "Class" in creditcard_df.columns

        quality_results = data_loader.validate_data_quality(creditcard_df, "creditcard")
        assert "imbalance_ratio" in quality_results
        assert quality_results["imbalance_ratio"] > 1

        engineered_df, scaler = feature_engineer.scale_numerical_features(
            creditcard_df, method="standard"
        )

        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            engineered_df, target_col="Class"
        )

        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X_train, y_train, method="smote"
        )

        model_trainer.train_logistic_regression(X_resampled, y_resampled)
        model_trainer.train_random_forest(X_resampled, y_resampled)

        for model_name, model in model_trainer.models.items():
            results = model_trainer.evaluate_model(model, X_test, y_test, model_name)

            assert 0 <= results["accuracy"] <= 1
            assert 0 <= results["f1_score"] <= 1

        comparison_df = model_trainer.compare_models(X_test, y_test)
        assert len(comparison_df) == 2

    def test_pipeline_with_different_imbalance_methods(self, pipeline_components):
        """Test pipeline with different class imbalance handling methods."""
        data_loader, feature_engineer, model_trainer = pipeline_components

        ecommerce_df = data_loader.load_ecommerce_data()

        engineered_df = feature_engineer.add_time_features(ecommerce_df)
        engineered_df = feature_engineer.encode_categorical_features(engineered_df)
        engineered_df, _ = feature_engineer.scale_numerical_features(engineered_df)

        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_df)

        methods = ["none", "smote", "undersample"]

        for method in methods:

            model_trainer.models = {}
            model_trainer.evaluation_results = {}

            X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
                X_train, y_train, method=method
            )

            model_trainer.train_logistic_regression(X_resampled, y_resampled)

            for model_name, model in model_trainer.models.items():
                results = model_trainer.evaluate_model(
                    model, X_test, y_test, f"{model_name}_{method}"
                )

                assert "f1_score" in results
                assert "precision" in results
                assert "recall" in results

    def test_pipeline_error_handling(self, pipeline_components):
        """Test pipeline error handling with invalid data."""
        data_loader, feature_engineer, model_trainer = pipeline_components

        with pytest.raises(FileNotFoundError):
            data_loader.load_ecommerce_data("nonexistent.csv")

        invalid_df = pd.DataFrame({"invalid_column": [1, 2, 3]})

        with pytest.raises(ValueError):
            model_trainer.prepare_data(invalid_df)

        X, y = pd.DataFrame({"feature": [1, 2, 3]}), pd.Series([0, 1, 0])

        with pytest.raises(ValueError):
            model_trainer.handle_class_imbalance(X, y, method="invalid")

    def test_pipeline_performance_benchmarks(self, pipeline_components):
        """Test pipeline performance with timing benchmarks."""
        import time

        data_loader, feature_engineer, model_trainer = pipeline_components

        start_time = time.time()
        ecommerce_df = data_loader.load_ecommerce_data()
        load_time = time.time() - start_time

        start_time = time.time()
        engineered_df = feature_engineer.add_time_features(ecommerce_df)
        engineered_df = feature_engineer.encode_categorical_features(engineered_df)
        feature_time = time.time() - start_time

        start_time = time.time()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_df)
        X_resampled, y_resampled, _ = model_trainer.handle_class_imbalance(
            X_train, y_train, method="smote"
        )
        model_trainer.train_logistic_regression(X_resampled, y_resampled)
        training_time = time.time() - start_time

        assert load_time < 10.0
        assert feature_time < 30.0
        assert training_time < 60.0

        print(f"Performance benchmarks:")
        print(f"Data loading: {load_time:.2f}s")
        print(f"Feature engineering: {feature_time:.2f}s")
        print(f"Model training: {training_time:.2f}s")

    def test_pipeline_reproducibility(self, pipeline_components):
        """Test that pipeline produces reproducible results."""
        data_loader, feature_engineer, model_trainer = pipeline_components

        ecommerce_df = data_loader.load_ecommerce_data()

        engineered_df = feature_engineer.add_time_features(ecommerce_df)
        engineered_df = feature_engineer.encode_categorical_features(engineered_df)
        engineered_df, _ = feature_engineer.scale_numerical_features(engineered_df)

        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_df)

        model_trainer.train_logistic_regression(X_train, y_train)
        predictions1 = model_trainer.models["logistic_regression"].predict(X_test)

        model_trainer.models = {}
        model_trainer.train_logistic_regression(X_train, y_train)
        predictions2 = model_trainer.models["logistic_regression"].predict(X_test)

        assert np.array_equal(predictions1, predictions2)
