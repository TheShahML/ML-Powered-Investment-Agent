# Makefile for ML Investment Bot - Local Docker Operations

# Variables
IMAGE_NAME := investment-bot
IMAGE_TAG := latest
CONTAINER_NAME := investment-bot-runner

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: help
help: ## Show this help message
	@echo "Investment Bot - Docker Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the Docker image
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: build-no-cache
build-no-cache: ## Build the Docker image without cache
	@echo "Building Docker image (no cache)..."
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: train
train: ## Train candidate models (3 horizons)
	@echo "Training candidate models..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/train_candidate.py

.PHONY: signals
signals: ## Generate daily trading signals
	@echo "Generating daily signals..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/generate_signals.py

.PHONY: rebalance
rebalance: ## Execute portfolio rebalance (if due)
	@echo "Executing rebalance..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/execute_rebalance_safe.py

.PHONY: rebalance-dry-run
rebalance-dry-run: ## Execute portfolio rebalance in dry-run mode
	@echo "Executing rebalance (DRY RUN)..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/execute_rebalance_safe.py --dry-run

.PHONY: health
health: ## Run health check
	@echo "Running health check..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/health_check.py

.PHONY: shell
shell: ## Open a shell inside the container
	@echo "Opening shell..."
	docker run --rm -it \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash

.PHONY: test
test: ## Run tests inside container
	@echo "Running tests..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd):/app" \
		-w /app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -m pytest tests/

.PHONY: deps-check
deps-check: ## Verify all dependencies are installed
	@echo "Checking dependencies..."
	docker run --rm \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -c "import xgboost; import pandas; import numpy; import alpaca_trade_api; print('✓ All dependencies OK')"

.PHONY: clean
clean: ## Remove Docker image
	@echo "Removing Docker image..."
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true

.PHONY: clean-all
clean-all: clean ## Remove Docker image and generated files
	@echo "Cleaning generated files..."
	rm -rf models/*.pkl data/signals/*.csv backtests/*.json logs/*.log

.PHONY: logs
logs: ## View container logs (if running)
	docker logs -f $(CONTAINER_NAME) || echo "No running container"

.PHONY: all
all: build deps-check ## Build image and verify dependencies
	@echo "✓ Build complete and dependencies verified"

.DEFAULT_GOAL := help
