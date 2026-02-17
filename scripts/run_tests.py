#!/usr/bin/env python3
"""
Test runner script for the fraud detection project.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Success: {description}")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for fraud detection project")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--quality", action="store_true", help="Run code quality checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to running unit tests if no specific option provided
    if not any([args.unit, args.integration, args.coverage, args.benchmark, 
                args.quality, args.security, args.all]):
        args.unit = True
    
    project_root = Path(__file__).parent.parent
    success = True
    
    # Change to project root
    import os
    os.chdir(project_root)
    
    # Code quality checks
    if args.quality or args.all:
        print("\n🔍 Running code quality checks...")
        
        # Lint with flake8
        if not run_command([
            "flake8", "src/", "tests/", "--count", "--select=E9,F63,F7,F82", 
            "--show-source", "--statistics"
        ], "Flake8 syntax check"):
            success = False
        
        # Check formatting with black
        if not run_command([
            "black", "--check", "--diff", "src/", "tests/"
        ], "Black formatting check"):
            success = False
        
        # Check import sorting with isort
        if not run_command([
            "isort", "--check-only", "--diff", "src/", "tests/"
        ], "Isort import check"):
            success = False
    
    # Security checks
    if args.security or args.all:
        print("\n🔒 Running security checks...")
        
        # Bandit security scan
        if not run_command([
            "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"
        ], "Bandit security scan"):
            success = False
        
        # Safety dependency check
        if not run_command([
            "safety", "check", "--json", "--output", "safety-report.json"
        ], "Safety dependency check"):
            success = False
    
    # Unit tests
    if args.unit or args.all:
        print("\n🧪 Running unit tests...")
        cmd = ["pytest", "tests/", "-m", "unit", "-v"]
        
        if args.coverage or args.all:
            cmd.extend([
                "--cov=src", 
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ])
        
        if not run_command(cmd, "Unit tests"):
            success = False
    
    # Integration tests
    if args.integration or args.all:
        print("\n🔗 Running integration tests...")
        cmd = ["pytest", "tests/", "-m", "integration", "-v"]
        
        if args.coverage or args.all:
            cmd.extend([
                "--cov=src", 
                "--cov-append",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ])
        
        if not run_command(cmd, "Integration tests"):
            success = False
    
    # Performance benchmarks
    if args.benchmark or args.all:
        print("\n📈 Running performance benchmarks...")
        if not run_command([
            "pytest", "tests/test_integration.py::TestIntegration::test_pipeline_performance_benchmarks",
            "-v", "--benchmark-only", "--benchmark-json=benchmark-results.json"
        ], "Performance benchmarks"):
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 All checks passed successfully!")
        
        if args.coverage or args.all:
            print("\n📊 Coverage reports generated:")
            print("  - Terminal: Already displayed above")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
        
        if args.benchmark or args.all:
            print("\n📈 Benchmark results saved to: benchmark-results.json")
        
        if args.security or args.all:
            print("\n🔒 Security reports generated:")
            print("  - Bandit: bandit-report.json")
            print("  - Safety: safety-report.json")
        
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
