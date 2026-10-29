.DEFAULT_GOAL := help

# PyTorch build: cu130 for NVIDIA GPUs (driver 580+), cpu for CI and CPU-only machines.
TORCH ?= cu130
# Training defaults; override on the command line, e.g. `make train DATA=... EPOCHS=50`.
DATA ?= data/processed/kitchen/data.yaml
MODEL ?= yolo26n.pt
EPOCHS ?= 100
IMGSZ ?= 640
BATCH ?= 16
DEVICE ?= 0
RUN_NAME ?= yolo26n-kitchen
# Extra Ultralytics arguments for `make train`, e.g. TRAIN_ARGS="seed=1 flipud=0.5".
TRAIN_ARGS ?=
# Parent folder of training run folders.
RUNS_DIR ?= $(CURDIR)/runs/detect
# MLflow store for training runs; Ultralytics' callback reads both variables.
export MLFLOW_TRACKING_URI ?= sqlite:///$(CURDIR)/runs/mlflow/mlflow.db
export MLFLOW_EXPERIMENT_NAME ?= objhist-detector

.PHONY: help bootstrap dev-web env-check train mlflow-ui format lint typecheck test test-gpu build check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Object history commands:\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Install Python (TORCH=cu130 or TORCH=cpu) and web dependencies
	uv sync --all-packages --extra $(TORCH)
	pnpm install

dev-web: ## Start the Next.js web app at http://localhost:3000
	pnpm --filter @objhist/web dev

env-check: ## Report PyTorch, CUDA, and Ultralytics versions; fail without CUDA
	uv run python -m objhist_detection.env_check --require-cuda

train: ## Fine-tune YOLO26 with the Ultralytics CLI; output goes to runs/detect/$(RUN_NAME)/ and MLflow
	uv run python -m objhist_detection.experiment_tracking init --new-run-dir $(RUNS_DIR)/$(RUN_NAME)
	YOLO_AUTOINSTALL=False uv run yolo detect train data=$(DATA) model=$(MODEL) epochs=$(EPOCHS) imgsz=$(IMGSZ) batch=$(BATCH) device=$(DEVICE) project=$(RUNS_DIR) name=$(RUN_NAME) $(TRAIN_ARGS)
	uv run python -m objhist_detection.experiment_tracking import-runs $(RUNS_DIR)/$(RUN_NAME)
	uv run python -m objhist_detection.experiment_tracking log-models --auto-best

mlflow-ui: ## Browse training runs in MLflow at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

format: ## Format Python and web code
	uv run ruff format .
	uv run ruff check --fix .
	pnpm -r exec prettier --write .

lint: ## Run Ruff and ESLint checks
	uv run ruff check .
	uv run ruff format --check .
	pnpm lint

typecheck: ## Type-check the web app
	pnpm typecheck

test: ## Run Python and web unit tests (GPU tests skip without CUDA)
	uv run pytest
	pnpm test

test-gpu: ## Run only the tests that need a CUDA device
	uv run pytest -m gpu

build: ## Build the web app for production
	pnpm build

check: lint typecheck test build ## Run all repository checks
