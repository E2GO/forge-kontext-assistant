#!/bin/bash
cd "$(dirname "$0")"

echo "Running pytest..."
python -m pytest tests/ -v --tb=short

# Run with coverage if needed
# python -m pytest tests/ --cov=ka_modules --cov-report=html --cov-report=term