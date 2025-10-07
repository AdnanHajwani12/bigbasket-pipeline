#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

# Tell Python to look in src/ for modules
export PYTHONPATH=src

# Run tests in tests/ directory, stop at first failure, concise output
pytest tests/ -q --maxfail=1 -o log_cli_level=INFO
