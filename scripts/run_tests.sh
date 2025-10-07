#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

# Run pytest on the tests/ directory, stop at first failure, show concise output
python -m pytest tests/ -q --maxfail=1 -o log_cli_level=INFO
