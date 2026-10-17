.PHONY: help setup-local up-local down-local ingest smoke-local test lint eval-retrieval eval-answers loadtest report gpu-up gpu-sync

PYTHON ?= python3
VENV ?= .venv

help:
	@echo "Available commands:"
	@echo "  make setup-local     - Set up venv and install dependencies"
	@echo "  make up-local        - Start local Docker stack (compose.local.yml)"
	@echo "  make down-local      - Stop local Docker stack"
	@echo "  make ingest          - Ingest sample documents into vector/keyword store"
	@echo "  make smoke-local     - Run local end-to-end smoke test"
	@echo "  make test            - Run pytest suite"
	@echo "  make lint            - Run ruff linter and mypy type checks"
	@echo "  make eval-retrieval  - Run retrieval evaluation (recall@k, MRR)"
	@echo "  make eval-answers    - Run answer evaluation with judge"
	@echo "  make loadtest        - Run async load test benchmark"
	@echo "  make report          - Generate benchmark and evaluation reports"
	@echo "  make gpu-up          - Start GPU Docker stack on AMD droplet"
	@echo "  make gpu-sync        - Commit and sync results from droplet"

setup-local:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv $(VENV) && uv pip install -e ".[app,eval,dev]"; \
	else \
		$(PYTHON) -m venv $(VENV) && ./$(VENV)/bin/pip install -e ".[app,eval,dev]"; \
	fi
	@echo "Local setup complete."

up-local:
	docker compose -f docker/compose.local.yml up -d

down-local:
	docker compose -f docker/compose.local.yml down

ingest:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python -m privatedocs.ingestion.indexer; \
	else \
		python -m privatedocs.ingestion.indexer; \
	fi

smoke-local:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python scripts/local/smoke_test.sh; \
	else \
		bash scripts/local/smoke_test.sh; \
	fi

test:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/pytest -v tests/; \
	else \
		pytest -v tests/; \
	fi

lint:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/ruff check . && ./$(VENV)/bin/mypy src/; \
	else \
		ruff check . && mypy src/; \
	fi

eval-retrieval:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python eval/run_retrieval_eval.py; \
	else \
		python eval/run_retrieval_eval.py; \
	fi

eval-answers:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python eval/run_answer_eval.py; \
	else \
		python eval/run_answer_eval.py; \
	fi

loadtest:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python benchmarks/load_test.py; \
	else \
		python benchmarks/load_test.py; \
	fi

report:
	@if [ -d "$(VENV)" ]; then \
		./$(VENV)/bin/python eval/report.py; \
	else \
		python eval/report.py; \
	fi

gpu-up:
	docker compose -f docker/compose.gpu.yml up -d

gpu-sync:
	bash scripts/gpu/04_sync_results.sh
