# Testing Guide for rtregistrationcalc

This document outlines the testing approach for the rtregistrationcalc package.

## Test Structure

Tests are organized by module and functionality:

1. `test_convert_matrix_to_euler.py` - Tests for the rotation matrix and Euler angle conversion functions
2. `test_extract_reg_matrix.py` - Tests for extracting matrices from registration objects
3. `test_extract_isocenter.py` - Tests for extracting isocenter coordinates from RTSS and RT Plan objects
4. `test_img_stack_functions.py` - Tests for image stack manipulation and center calculation
5. `test_compute_6dof.py` - Tests for calculating 6DOF corrections from registration, RTSS, and plan data
6. `conftest.py` - Common test fixtures shared across test modules

## Running the Tests

To run all tests:

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest test_convert_matrix_to_euler.py
```

To run tests with verbose output:

```bash
python -m pytest -v
```

To run tests and generate a coverage report:

```bash
python -m pytest --cov=rtregistrationcalc
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `create_temp_directory` - Creates a temporary directory for test files
- `create_mock_ct_dataset` - Creates a mock CT DICOM dataset
- `create_mock_registration_dataset` - Creates a mock registration dataset with transformation matrix

## Writing New Tests

When adding new functionality to the codebase, follow these guidelines for test creation:

1. Create a new test file if testing a new module, or add tests to an existing file if extending functionality
2. Write test functions with clear, descriptive names that indicate what they're testing
3. Use pytest fixtures for setup and teardown
4. Test both normal operation and edge cases/error conditions
5. For DICOM object manipulation, create minimal mock datasets instead of using real DICOM files
6. Follow the naming pattern: `test_<function_name>_<scenario>`

## Test Coverage

Aim for high test coverage, particularly for the core computational functions. At minimum, tests should verify:

1. Correct behavior for typical inputs
2. Proper error handling for invalid inputs
3. Edge cases (e.g., singular matrices, empty datasets)
4. Round-trip conversions where applicable (e.g., matrix → angles → matrix)

## Dependencies

The tests require:

- pytest
- pytest-cov (optional, for coverage reports)
- numpy
- pydicom

These can be installed via Poetry:

```bash
poetry add --dev pytest pytest-cov
```

Or with pip:

```bash
pip install pytest pytest-cov
```