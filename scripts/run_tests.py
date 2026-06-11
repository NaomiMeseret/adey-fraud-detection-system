#!/usr/bin/env python3
"""
Test runner script for the fraud detection project.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
UNIT_TEST_FILES = [
    "tests/test_data_loader.py",
    "tests/test_feature_engineering.py",
    "tests/test_model_trainer.py",
]
INTEGRATION_TEST_FILES = ["tests/test_integration.py"]
QUALITY_PATHS = ["src/", "tests/", "scripts/"]


def python_files(paths):
    """Return Python files under the provided paths."""
    files = []
    for path in paths:
        root = Path(path)
        if root.is_file() and root.suffix == ".py":
            files.append(str(root))
        elif root.is_dir():
            files.extend(str(file_path) for file_path in root.rglob("*.py"))
    return sorted(files)


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Failed: {description}")
        return False

    print(f"Success: {description}")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for fraud detection project"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--quality", action="store_true", help="Run code quality checks"
    )
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not any(
        [
            args.unit,
            args.integration,
            args.coverage,
            args.benchmark,
            args.quality,
            args.security,
            args.all,
        ]
    ):
        args.unit = True

    project_root = Path(__file__).parent.parent
    success = True

    import os

    os.chdir(project_root)

    if args.quality or args.all:
        print("\nRunning code quality checks...")

        if not run_command(
            [
                PYTHON,
                "-m",
                "flake8",
                *QUALITY_PATHS,
                "--count",
                "--select=E9,F63,F7,F82",
                "--show-source",
                "--statistics",
            ],
            "Flake8 syntax check",
        ):
            success = False

        for file_path in python_files(QUALITY_PATHS):
            if not run_command(
                [PYTHON, "-m", "black", "--check", "--diff", file_path],
                f"Black formatting check ({file_path})",
            ):
                success = False

        if not run_command(
            [
                PYTHON,
                "-m",
                "isort",
                "--profile",
                "black",
                "--check-only",
                "--diff",
                *QUALITY_PATHS,
            ],
            "Isort import check",
        ):
            success = False

    if args.security or args.all:
        print("\nRunning security checks...")

        if not run_command(
            [
                PYTHON,
                "-m",
                "bandit",
                "-r",
                "src/",
                "-f",
                "json",
                "-o",
                "bandit-report.json",
            ],
            "Bandit security scan",
        ):
            success = False

        if not run_command(
            [
                PYTHON,
                "-m",
                "safety",
                "check",
                "--output",
                "json",
            ],
            "Safety dependency check",
        ):
            success = False

    if args.unit or args.all:
        print("\nRunning unit tests...")
        cmd = [PYTHON, "-m", "pytest", *UNIT_TEST_FILES, "-v"]

        if args.coverage or args.all:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml",
                ]
            )

        if not run_command(cmd, "Unit tests"):
            success = False

    if args.integration or args.all:
        print("\nRunning integration tests...")
        cmd = [PYTHON, "-m", "pytest", *INTEGRATION_TEST_FILES, "-v"]

        if args.coverage or args.all:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml",
                ]
            )

        if not run_command(cmd, "Integration tests"):
            success = False

    if args.benchmark or args.all:
        print("\nRunning performance benchmarks...")
        if not run_command(
            [
                PYTHON,
                "-m",
                "pytest",
                "tests/test_integration.py::TestIntegration::test_pipeline_performance_benchmarks",
                "-v",
            ],
            "Performance benchmarks",
        ):
            success = False

    print("\n" + "=" * 60)
    if success:
        print("All checks passed successfully.")

        if args.coverage or args.all:
            print("\nCoverage reports generated:")
            print("  - Terminal: Already displayed above")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")

        if args.benchmark or args.all:
            print("\nPerformance benchmark test completed.")

        if args.security or args.all:
            print("\nSecurity reports generated:")
            print("  - Bandit: bandit-report.json")
            print("  - Safety: JSON output displayed in terminal")

        return 0
    else:
        print("Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
