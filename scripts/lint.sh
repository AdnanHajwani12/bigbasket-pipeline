#!/usr/bin/env bash
set -e
ruff src tests
black --check .