# Source code package (Role 1 — project content: shared definitions for fraud detection)

# --- E-commerce fraud dataset (Fraud_Data.csv) ---
ECOMMERCE_RAW_FILE = "Fraud_Data.csv"
ECOMMERCE_TARGET_COLUMN = "class"
ECOMMERCE_FRAUD_LABEL = 1
ECOMMERCE_LEGIT_LABEL = 0

# --- Credit card fraud dataset (creditcard.csv) ---
CREDITCARD_RAW_FILE = "creditcard.csv"
CREDITCARD_TARGET_COLUMN = "Class"
CREDITCARD_FRAUD_LABEL = 1
CREDITCARD_LEGIT_LABEL = 0


def is_fraud(label: int, dataset: str = "ecommerce") -> bool:
    """Return True if label is fraud (1) for the given dataset."""
    return label == (ECOMMERCE_FRAUD_LABEL if dataset == "ecommerce" else CREDITCARD_FRAUD_LABEL)


def target_column(dataset: str = "ecommerce") -> str:
    """Return the target column name for the given dataset."""
    return ECOMMERCE_TARGET_COLUMN if dataset == "ecommerce" else CREDITCARD_TARGET_COLUMN
