.PHONY: all clean docker install dev lint format test help

# Default architecture
SASS_ARCH ?= sm_80

help:
	@echo "Available targets:"
	@echo "  make all       - Build SASS files for $(SASS_ARCH)"
	@echo "  make docker    - Build using Docker"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make install   - Install package in editable mode"
	@echo "  make dev       - Install with development dependencies"
	@echo "  make lint      - Run code linters"
	@echo "  make format    - Format code with black"
	@echo "  make test      - Run tests"
	@echo ""
	@echo "Variables:"
	@echo "  SASS_ARCH=sm_XX - Set target architecture (default: sm_80)"

all:
	@echo "Building for $(SASS_ARCH)..."
	@SASS_ARCH=$(SASS_ARCH) bash scripts/build.sh

docker:
	@bash scripts/docker-build.sh

clean:
	@bash scripts/clean.sh
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

install:
	@echo "Installing nvidia-sass-lessons package..."
	@pip install -e .

dev:
	@echo "Installing development dependencies..."
	@pip install -e ".[dev]"

lint:
	@echo "Running linters..."
	@which ruff > /dev/null 2>&1 && ruff check . || echo "Ruff not installed, skipping"
	@which mypy > /dev/null 2>&1 && mypy sass_lessons || echo "Mypy not installed, skipping"

format:
	@echo "Formatting code..."
	@which black > /dev/null 2>&1 && black sass_lessons lesson_*/*.py || echo "Black not installed, run 'make dev' first"
	@which ruff > /dev/null 2>&1 && ruff check --fix . || echo "Ruff not installed, skipping"

test:
	@echo "Running tests..."
	@which pytest > /dev/null 2>&1 && pytest || echo "Pytest not installed, run 'make dev' first"