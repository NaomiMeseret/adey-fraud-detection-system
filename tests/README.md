# Testing Suite

This directory contains comprehensive tests for the fraud detection project, including unit tests, integration tests, and quality assurance checks.

## Test Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Pytest configuration and shared fixtures
├── test_data_loader.py       # Tests for data loading functionality
├── test_feature_engineering.py  # Tests for feature engineering
├── test_model_trainer.py    # Tests for model training and evaluation
└── test_integration.py      # End-to-end integration tests
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods in isolation
- Fast execution
- Mock external dependencies
- Focus on business logic correctness

### Integration Tests (`@pytest.mark.integration`)
- Test multiple components working together
- Use real data (generated for testing)
- Slower execution but more realistic
- Validate end-to-end workflows

## Running Tests

### Quick Start

```bash
# Run all unit tests
pytest tests/ -m "unit"

# Run all integration tests
pytest tests/ -m "integration"

# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Using the Test Runner Script

The project includes a comprehensive test runner script:

```bash
# Run unit tests only
python scripts/run_tests.py --unit

# Run integration tests only
python scripts/run_tests.py --integration

# Run all checks (unit, integration, quality, security, benchmarks)
python scripts/run_tests.py --all

# Run with coverage report
python scripts/run_tests.py --coverage

# Run performance benchmarks
python scripts/run_tests.py --benchmark

# Run code quality checks
python scripts/run_tests.py --quality

# Run security checks
python scripts/run_tests.py --security
```

### Test Categories Explained

#### Unit Tests
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (seconds)
- **Dependencies**: Mocked
- **Examples**:
  - Data loading with valid/invalid files
  - Feature engineering functions
  - Model training with sample data

#### Integration Tests
- **Purpose**: Test complete workflows
- **Speed**: Slower (minutes)
- **Dependencies**: Real (generated) data
- **Examples**:
  - Complete fraud detection pipeline
  - End-to-end data processing
  - Model comparison workflows

#### Quality Checks
- **Linting**: Code style and syntax
- **Formatting**: Black and isort compliance
- **Type checking**: Mypy static analysis
- **Security**: Bandit and Safety scans

#### Performance Tests
- **Benchmarks**: Pipeline execution time
- **Memory usage**: Resource consumption
- **Scalability**: Performance with different data sizes

## Test Data

Tests use generated synthetic data to ensure:
- Reproducibility
- No dependency on external files
- Controlled test scenarios
- Privacy compliance

### Sample Data Generation

Integration tests create realistic test data including:
- E-commerce transactions with fraud patterns
- Credit card transactions with PCA features
- IP address to country mappings
- Various class imbalance scenarios

## Coverage

The test suite aims for high coverage across:
- Core business logic
- Edge cases and error handling
- Data validation
- Model training pipelines

Coverage reports are generated in:
- **Terminal**: Immediate feedback
- **HTML**: Detailed browser view (`htmlcov/index.html`)
- **XML**: CI/CD integration (`coverage.xml`)

## CI/CD Integration

### GitHub Actions Workflows

1. **Quick Unit Tests** (`unittests.yml`):
   - Runs on every push/PR
   - Multiple Python versions
   - Fast feedback

2. **Full CI/CD Pipeline** (`ci-cd.yml`):
   - Comprehensive quality checks
   - Security scanning
   - Performance benchmarks
   - Documentation generation
   - Artifact uploads

### Quality Gates

The pipeline enforces:
- ✅ All tests must pass
- ✅ Code quality standards
- ✅ Security requirements
- ✅ Performance benchmarks

## Writing New Tests

### Unit Test Template

```python
import pytest
from src.module import ClassOrFunction

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = ...
        expected = ...
        
        # Act
        result = ClassOrFunction(input_data)
        
        # Assert
        assert result == expected
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty data, invalid inputs, etc.
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        # Test that appropriate exceptions are raised
        with pytest.raises(ExpectedException):
            ClassOrFunction(invalid_input)
```

### Integration Test Template

```python
@pytest.mark.integration
class TestNewWorkflow:
    def test_complete_workflow(self, pipeline_components):
        """Test complete workflow."""
        data_loader, feature_engineer, model_trainer = pipeline_components
        
        # Load data
        df = data_loader.load_data()
        
        # Process data
        processed_df = feature_engineer.process(df)
        
        # Train model
        model = model_trainer.train(processed_df)
        
        # Validate results
        assert model is not None
        assert len(model_trainer.evaluation_results) > 0
```

## Best Practices

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow Arrange-Act-Assert pattern
- Add docstrings explaining test purpose

### Test Data
- Use fixtures for shared test data
- Generate data programmatically
- Include edge cases and error scenarios
- Keep test data small and focused

### Assertions
- Use specific assertions (assert_equal, assert_in, etc.)
- Include helpful error messages
- Test both positive and negative cases
- Validate types and ranges

### Mocking
- Mock external dependencies
- Use consistent mock objects
- Verify mock interactions
- Keep mocks simple and focused

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test file
pytest tests/test_data_loader.py -v

# Run specific test class
pytest tests/test_data_loader.py::TestDataLoader -v

# Run specific test method
pytest tests/test_data_loader.py::TestDataLoader::test_load_ecommerce_data_success -v

# Run with debugging
pytest tests/test_data_loader.py::TestDataLoader::test_load_ecommerce_data_success -v -s --pdb
```

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **Missing Data**: Check that test fixtures create required data
3. **Timing Issues**: Use appropriate timeouts for integration tests
4. **Randomness**: Set random seeds for reproducible tests

## Continuous Improvement

### Adding New Tests
1. Write tests for new features
2. Ensure coverage remains high
3. Add appropriate markers (unit/integration)
4. Update documentation

### Monitoring
- Track test execution time
- Monitor coverage trends
- Review flaky tests
- Update tests as code evolves

### Performance
- Profile slow tests
- Optimize test data generation
- Use parallel execution where appropriate
- Cache expensive operations

## Troubleshooting

### Common Test Failures

1. **Dependency Issues**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Path Issues**:
   ```bash
   # Run from project root
   cd /path/to/project
   pytest tests/
   ```

3. **Environment Issues**:
   ```bash
   # Use virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

### Getting Help

- Check test output for specific error messages
- Review failing test code and expected behavior
- Consult project documentation
- Reach out to the development team

## Contributing

When contributing to the test suite:

1. Follow existing patterns and conventions
2. Add appropriate test markers
3. Update documentation as needed
4. Ensure all tests pass before submitting
5. Maintain high code coverage

Thank you for helping maintain the quality and reliability of the fraud detection system!
