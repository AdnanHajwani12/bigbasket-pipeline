#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

# Make Python aware of src/ folder for imports
export PYTHONPATH=src

# Run pytest on the tests/ directory, stop at first failure, show concise logs
pytest tests/ -q --maxfail=1 -o log_cli_level=INFO
