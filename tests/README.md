# Tests

The test suite checks the core fraud detection workflow: loading data, creating features, training models, evaluating results, and running the main pipeline end to end.

## Test Files

```text
tests/
├── conftest.py
├── test_data_loader.py
├── test_feature_engineering.py
├── test_model_trainer.py
└── test_integration.py
```

## Running Tests

Run all tests:

```bash
python -m pytest -q
```

Run unit-focused tests:

```bash
python scripts/run_tests.py --unit
```

Run integration tests:

```bash
python scripts/run_tests.py --integration
```

Run coverage:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

Run code quality checks:

```bash
python scripts/run_tests.py --quality
```

## Test Data

The tests create synthetic data instead of depending on large external files. This keeps the test suite fast, reproducible, and safe to run in CI.

The integration tests cover:

- E-commerce fraud pipeline
- Credit card fraud pipeline
- Class imbalance handling
- Error handling
- Pipeline timing
- Reproducibility

## Current Status

The local suite currently passes with 61 tests and 96% source coverage.
