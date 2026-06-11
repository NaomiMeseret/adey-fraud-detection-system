"""
Model training utilities for fraud detection project.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation for fraud detection."""

    def __init__(self, random_state: int = 42):
        """Initialize ModelTrainer with random state."""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "class",
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Proportion of data for testing
            stratify: Whether to stratify the split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column not found: {target_col}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        id_cols = [
            "user_id",
            "device_id",
            "ip_address",
            "ip_int",
            "signup_time",
            "purchase_time",
        ]
        X = X.drop(columns=[col for col in id_cols if col in X.columns])

        unsupported_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if unsupported_cols:
            logger.warning("Dropping non-numeric model features: %s", unsupported_cols)
            X = X.drop(columns=unsupported_cols)

        bool_cols = X.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)

        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def handle_class_imbalance(
        self, X_train: pd.DataFrame, y_train: pd.Series, method: str = "smote"
    ) -> Tuple[pd.DataFrame, pd.Series, Any]:
        """
        Handle class imbalance in training data.

        Args:
            X_train: Training features
            y_train: Training target
            method: Method to handle imbalance ('smote', 'undersample', 'none')

        Returns:
            Tuple of (resampled X_train, resampled y_train, fitted sampler)
        """
        if method == "none":
            return X_train, y_train, None

        if method == "smote":
            sampler = SMOTE(random_state=self.random_state)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown imbalance handling method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        logger.info(f"Class imbalance handled using {method}")
        logger.info(f"Original distribution: {y_train.value_counts().to_dict()}")
        logger.info(
            f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}"
        )

        return X_resampled, y_resampled, sampler

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for LogisticRegression

        Returns:
            Trained LogisticRegression model
        """
        params = {
            "random_state": self.random_state,
            "max_iter": 1000,
            "class_weight": "balanced",
        }
        params.update(kwargs)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        self.models["logistic_regression"] = model
        logger.info("Trained Logistic Regression model")

        return model

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Trained RandomForestClassifier model
        """
        params = {
            "n_estimators": 100,
            "random_state": self.random_state,
            "class_weight": "balanced",
        }
        params.update(kwargs)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        self.models["random_forest"] = model
        logger.info("Trained Random Forest model")

        return model

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for XGBClassifier

        Returns:
            Trained XGBClassifier model
        """
        positive_count = y_train.sum()
        negative_count = len(y_train) - positive_count

        params = {
            "n_estimators": 100,
            "random_state": self.random_state,
            "eval_metric": "logloss",
            "scale_pos_weight": negative_count / max(positive_count, 1),
        }
        params.update(kwargs)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        self.models["xgboost"] = model
        logger.info("Trained XGBoost model")

        return model

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for results

        Returns:
            Dictionary with evaluation metrics
        """

        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        results = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

        if y_pred_proba is not None:
            results["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            results["average_precision"] = average_precision_score(y_test, y_pred_proba)

        self.evaluation_results[model_name] = results

        logger.info(
            f"Evaluated {model_name}: F1={results['f1_score']:.3f}, "
            f"Precision={results['precision']:.3f}, Recall={results['recall']:.3f}"
        )

        return results

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            "cv_scores": scores.tolist(),
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scoring_metric": scoring,
        }

        logger.info(
            f"Cross-validation results - Mean {scoring}: {results['mean_score']:.3f} "
            f"(±{results['std_score']:.3f})"
        )

        return results

    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare all trained models.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for model_name, model in self.models.items():
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Accuracy": results["accuracy"],
                        "Precision": results["precision"],
                        "Recall": results["recall"],
                        "F1 Score": results["f1_score"],
                        "ROC AUC": results.get("roc_auc", np.nan),
                        "Avg Precision": results.get("average_precision", np.nan),
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("F1 Score", ascending=False)

        logger.info("Model comparison completed")
        return comparison_df

    def get_feature_importance(
        self, model: Any, feature_names: List[str], model_name: str = "model"
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        logger.info(f"Extracted feature importance for {model_name}")
        return importance_df
