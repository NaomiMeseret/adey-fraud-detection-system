"""
Unit tests for model_trainer module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.model_trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            weights=[0.9, 0.1],  # Imbalanced dataset
            random_state=42
        )
        
        # Convert to DataFrame with meaningful column names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='class')
        
        return X_df, y_series
    
    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer(random_state=42)
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=123)
        assert trainer.random_state == 123
        assert trainer.models == {}
        assert trainer.scalers == {}
        assert trainer.evaluation_results == {}
    
    def test_prepare_data(self, model_trainer, sample_classification_data):
        """Test data preparation for training."""
        X, y = sample_classification_data
        df = pd.concat([X, y], axis=1)
        
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(df)
        
        # Check shapes
        assert X_train.shape[0] == 800  # 80% of 1000
        assert X_test.shape[0] == 200   # 20% of 1000
        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]
        
        # Check that target column is removed from features
        assert 'class' not in X_train.columns
        assert 'class' not in X_test.columns
        
        # Check stratification
        train_fraud_ratio = y_train.mean()
        test_fraud_ratio = y_test.mean()
        assert abs(train_fraud_ratio - test_fraud_ratio) < 0.05
    
    def test_prepare_data_no_stratify(self, model_trainer, sample_classification_data):
        """Test data preparation without stratification."""
        X, y = sample_classification_data
        df = pd.concat([X, y], axis=1)
        
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            df, stratify=False
        )
        
        # Check shapes
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200
    
    def test_prepare_data_custom_test_size(self, model_trainer, sample_classification_data):
        """Test data preparation with custom test size."""
        X, y = sample_classification_data
        df = pd.concat([X, y], axis=1)
        
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            df, test_size=0.3
        )
        
        # Check shapes
        assert X_train.shape[0] == 700  # 70% of 1000
        assert X_test.shape[0] == 300   # 30% of 1000
    
    def test_handle_class_imbalance_smote(self, model_trainer, sample_classification_data):
        """Test class imbalance handling with SMOTE."""
        X, y = sample_classification_data
        
        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X, y, method='smote'
        )
        
        # Check that sampler is returned
        assert sampler is not None
        
        # Check that classes are balanced
        class_counts = y_resampled.value_counts()
        assert class_counts[0] == class_counts[1]
        
        # Check that feature count is preserved
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_handle_class_imbalance_undersample(self, model_trainer, sample_classification_data):
        """Test class imbalance handling with undersampling."""
        X, y = sample_classification_data
        
        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X, y, method='undersample'
        )
        
        # Check that sampler is returned
        assert sampler is not None
        
        # Check that classes are balanced
        class_counts = y_resampled.value_counts()
        assert class_counts[0] == class_counts[1]
        
        # Check that minority class size is preserved
        original_minority_count = y.value_counts().min()
        assert class_counts[1] == original_minority_count
    
    def test_handle_class_imbalance_none(self, model_trainer, sample_classification_data):
        """Test class imbalance handling with no method."""
        X, y = sample_classification_data
        
        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X, y, method='none'
        )
        
        # Check that no sampler is returned
        assert sampler is None
        
        # Check that data is unchanged
        assert X_resampled.equals(X)
        assert y_resampled.equals(y)
    
    def test_handle_class_imbalance_invalid_method(self, model_trainer, sample_classification_data):
        """Test class imbalance handling with invalid method."""
        X, y = sample_classification_data
        
        with pytest.raises(ValueError, match="Unknown imbalance handling method"):
            model_trainer.handle_class_imbalance(X, y, method='invalid')
    
    def test_train_logistic_regression(self, model_trainer, sample_classification_data):
        """Test training Logistic Regression model."""
        X, y = sample_classification_data
        
        model = model_trainer.train_logistic_regression(X, y)
        
        # Check that model is trained and stored
        assert model is not None
        assert 'logistic_regression' in model_trainer.models
        assert model_trainer.models['logistic_regression'] == model
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_train_logistic_regression_custom_params(self, model_trainer, sample_classification_data):
        """Test training Logistic Regression with custom parameters."""
        X, y = sample_classification_data
        
        model = model_trainer.train_logistic_regression(
            X, y, max_iter=500, C=0.1
        )
        
        # Check that custom parameters were used
        assert model.max_iter == 500
        assert model.C == 0.1
    
    def test_train_random_forest(self, model_trainer, sample_classification_data):
        """Test training Random Forest model."""
        X, y = sample_classification_data
        
        model = model_trainer.train_random_forest(X, y)
        
        # Check that model is trained and stored
        assert model is not None
        assert 'random_forest' in model_trainer.models
        assert isinstance(model, RandomForestClassifier)
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_train_random_forest_custom_params(self, model_trainer, sample_classification_data):
        """Test training Random Forest with custom parameters."""
        X, y = sample_classification_data
        
        model = model_trainer.train_random_forest(
            X, y, n_estimators=50, max_depth=5
        )
        
        # Check that custom parameters were used
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_train_xgboost(self, model_trainer, sample_classification_data):
        """Test training XGBoost model."""
        X, y = sample_classification_data
        
        model = model_trainer.train_xgboost(X, y)
        
        # Check that model is trained and stored
        assert model is not None
        assert 'xgboost' in model_trainer.models
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_train_xgboost_custom_params(self, model_trainer, sample_classification_data):
        """Test training XGBoost with custom parameters."""
        X, y = sample_classification_data
        
        model = model_trainer.train_xgboost(
            X, y, n_estimators=50, learning_rate=0.1
        )
        
        # Check that custom parameters were used
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
    
    def test_evaluate_model(self, model_trainer, sample_classification_data):
        """Test model evaluation."""
        X, y = sample_classification_data
        
        # Train a simple model
        model = model_trainer.train_logistic_regression(X, y)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Retrain on training data
        model.fit(X_train, y_train)
        
        # Evaluate
        results = model_trainer.evaluate_model(model, X_test, y_test, "test_model")
        
        # Check that all expected metrics are present
        expected_metrics = [
            'model_name', 'accuracy', 'precision', 'recall', 'f1_score',
            'confusion_matrix', 'classification_report', 'roc_auc', 'average_precision'
        ]
        
        for metric in expected_metrics:
            assert metric in results
        
        # Check metric ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        assert 0 <= results['roc_auc'] <= 1
        assert 0 <= results['average_precision'] <= 1
        
        # Check that results are stored
        assert 'test_model' in model_trainer.evaluation_results
    
    def test_cross_validate_model(self, model_trainer, sample_classification_data):
        """Test model cross-validation."""
        X, y = sample_classification_data
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Cross-validate
        results = model_trainer.cross_validate_model(
            model, X, y, cv_folds=3, scoring='f1'
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['cv_scores', 'mean_score', 'std_score', 'scoring_metric']
        
        for metric in expected_metrics:
            assert metric in results
        
        # Check that we have the right number of CV scores
        assert len(results['cv_scores']) == 3
        
        # Check scoring metric
        assert results['scoring_metric'] == 'f1'
    
    def test_compare_models(self, model_trainer, sample_classification_data):
        """Test model comparison."""
        X, y = sample_classification_data
        
        # Split data
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            pd.concat([X, y], axis=1)
        )
        
        # Train multiple models
        model_trainer.train_logistic_regression(X_train, y_train)
        model_trainer.train_random_forest(X_train, y_train)
        
        # Evaluate models
        for model_name, model in model_trainer.models.items():
            model_trainer.evaluate_model(model, X_test, y_test, model_name)
        
        # Compare models
        comparison_df = model_trainer.compare_models(X_test, y_test)
        
        # Check that comparison DataFrame has the right structure
        expected_columns = [
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Avg Precision'
        ]
        
        for col in expected_columns:
            assert col in comparison_df.columns
        
        # Check that we have results for all trained models
        assert len(comparison_df) == 2
        
        # Check that DataFrame is sorted by F1 Score
        f1_scores = comparison_df['F1 Score'].values
        assert all(f1_scores[i] >= f1_scores[i+1] for i in range(len(f1_scores)-1))
    
    def test_get_feature_importance_random_forest(self, model_trainer, sample_classification_data):
        """Test feature importance extraction for Random Forest."""
        X, y = sample_classification_data
        
        # Train Random Forest
        model = model_trainer.train_random_forest(X, y)
        
        # Get feature importance
        importance_df = model_trainer.get_feature_importance(
            model, X.columns.tolist(), 'random_forest'
        )
        
        # Check that importance DataFrame has the right structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        
        # Check that importance values are non-negative
        assert all(importance_df['importance'] >= 0)
        
        # Check that DataFrame is sorted by importance
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_get_feature_importance_logistic_regression(self, model_trainer, sample_classification_data):
        """Test feature importance extraction for Logistic Regression."""
        X, y = sample_classification_data
        
        # Train Logistic Regression
        model = model_trainer.train_logistic_regression(X, y)
        
        # Get feature importance
        importance_df = model_trainer.get_feature_importance(
            model, X.columns.tolist(), 'logistic_regression'
        )
        
        # Check that importance DataFrame has the right structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        
        # Check that importance values are non-negative (absolute coefficients)
        assert all(importance_df['importance'] >= 0)
    
    def test_get_feature_importance_no_importance(self, model_trainer, sample_classification_data):
        """Test feature importance extraction for model without importance."""
        X, y = sample_classification_data
        
        # Create a mock model without feature importance
        mock_model = MagicMock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        # Get feature importance
        importance_df = model_trainer.get_feature_importance(
            mock_model, X.columns.tolist(), 'mock_model'
        )
        
        # Should return empty DataFrame
        assert importance_df.empty
    
    def test_integration_full_pipeline(self, model_trainer, sample_classification_data):
        """Test integration of full training and evaluation pipeline."""
        X, y = sample_classification_data
        df = pd.concat([X, y], axis=1)
        
        # Prepare data
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(df)
        
        # Handle imbalance
        X_resampled, y_resampled, sampler = model_trainer.handle_class_imbalance(
            X_train, y_train, method='smote'
        )
        
        # Train models
        model_trainer.train_logistic_regression(X_resampled, y_resampled)
        model_trainer.train_random_forest(X_resampled, y_resampled)
        
        # Evaluate models
        for model_name, model in model_trainer.models.items():
            results = model_trainer.evaluate_model(model, X_test, y_test, model_name)
            assert results['f1_score'] >= 0
        
        # Compare models
        comparison_df = model_trainer.compare_models(X_test, y_test)
        assert len(comparison_df) == 2
        
        # Get feature importance for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = model_trainer.models[best_model_name]
        importance_df = model_trainer.get_feature_importance(
            best_model, X_test.columns.tolist(), best_model_name
        )
        
        assert not importance_df.empty
        assert len(importance_df) == X_test.shape[1]
