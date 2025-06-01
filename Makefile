.PHONY: clean lint format test coverage docs build install dev-install

# Default target
all: lint test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Lint code
lint:
	flake8 .
	black --check .
	isort --check-only --profile black .

# Format code
format:
	black .
	isort --profile black .

# Run tests
test:
	pytest

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=term --cov-report=html

# Build documentation
docs:
	sphinx-build -b html docs/source docs/build

# Build package
build: clean
	python setup.py sdist bdist_wheel

# Install package
install:
	pip install .

# Install package in development mode
dev-install:
	pip install -e ".[dev]"

# Train model
train:
	python train.py

# Evaluate model
evaluate:
	python evaluation.py

# Generate submission
submission:
	python generate_submission.py

# Optimize model for mobile
optimize:
	python quantize.py