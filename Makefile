# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Health Insurance AI Platform
# Common developer commands. Run: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install setup data pipeline train test lint format docker-build \
        docker-up docker-down api clean

PYTHON     := python
PIP        := pip
SRC        := src
TESTS      := tests
DATA_DIR   := data/synthetic
PROC_DIR   := data/processed
MODEL_DIR  := models/saved

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Health Insurance AI Platform — Developer Commands"
	@echo "─────────────────────────────────────────────────"
	@echo "  make install       Install Python dependencies"
	@echo "  make setup         Full setup: install + spacy model + dirs"
	@echo "  make data          Generate synthetic training data"
	@echo "  make pipeline      Run full data pipeline (ingest → features)"
	@echo "  make train         Train fraud detection model"
	@echo "  make test          Run full test suite with coverage"
	@echo "  make test-fast     Run unit tests only (no model training)"
	@echo "  make lint          Run flake8 + black + isort checks"
	@echo "  make format        Auto-format code with black + isort"
	@echo "  make api           Start FastAPI server (reload mode)"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start full local stack (docker compose)"
	@echo "  make docker-down   Stop local stack"
	@echo "  make clean         Remove generated files"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup: install
	$(PYTHON) -m spacy download en_core_web_sm
	mkdir -p $(DATA_DIR) $(PROC_DIR) $(MODEL_DIR) models/reports logs

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	@echo "Generating synthetic health insurance data..."
	$(PYTHON) $(SRC)/ingestion/generate_synthetic_data.py

pipeline: data
	@echo "Running full data pipeline..."
	$(PYTHON) $(SRC)/ingestion/pipeline_runner.py --source synthetic --env dev

# ── Model Training ────────────────────────────────────────────────────────────
train: pipeline
	@echo "Training fraud detection model..."
	$(PYTHON) $(SRC)/models/fraud_model.py \
		--data $(PROC_DIR)/features_dev.parquet \
		--cv

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest $(TESTS)/ \
		-v \
		--tb=short \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=70

test-fast:
	pytest $(TESTS)/ -v --tb=short -m "not slow" -x

test-api:
	pytest $(TESTS)/ -v --tb=short -m "api"

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	flake8 $(SRC)/ $(TESTS)/ \
		--max-line-length=120 \
		--ignore=E203,W503,E501 \
		--exclude=__pycache__,.venv
	black --check $(SRC)/ $(TESTS)/
	isort --check-only $(SRC)/ $(TESTS)/

format:
	black $(SRC)/ $(TESTS)/
	isort $(SRC)/ $(TESTS)/

# ── API ───────────────────────────────────────────────────────────────────────
api:
	uvicorn src.api.main:app \
		--reload \
		--host 0.0.0.0 \
		--port 8000 \
		--log-level info

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t health-insurance-ai:latest .

docker-up:
	docker compose --profile dev up -d
	@echo "Services started:"
	@echo "  API:     http://localhost:8000/docs"
	@echo "  MLflow:  http://localhost:5000"
	@echo "  PgAdmin: http://localhost:5050"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

# ── Notebooks ─────────────────────────────────────────────────────────────────
notebooks:
	jupyter notebook notebooks/ --port=8888 --no-browser

# ── Monitoring ────────────────────────────────────────────────────────────────
monitor:
	$(PYTHON) $(SRC)/monitoring/model_monitor.py \
		--reference $(PROC_DIR)/features_dev.parquet \
		--current   $(PROC_DIR)/features_dev.parquet

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	find . -type f -name "*.pyo" -delete 2>/dev/null; true
	rm -rf htmlcov/ .coverage coverage.xml .pytest_cache/
	@echo "Cleaned."

clean-data:
	rm -f $(DATA_DIR)/*.csv
	rm -f $(PROC_DIR)/*.parquet
	@echo "Data files removed."

clean-models:
	rm -f $(MODEL_DIR)/*.joblib
	rm -f $(MODEL_DIR)/*.json
	rm -f models/reports/*.png models/reports/*.json
	@echo "Model artifacts removed."
