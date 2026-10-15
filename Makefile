.PHONY: install test test-docker report chunk-report retrieval-report tool-report holiday-report live-smoke download-models run up down

install:
	pip install -r requirements-dev.txt

# Tests that need OCR (tesseract + poppler) are skipped if those tools are
# missing locally; `make test-docker` runs everything inside the image.
test:
	python -m pytest

test-docker:
	docker compose run --rm --no-deps -v "$(CURDIR):/app" app sh -c "pip install -q -r requirements-dev.txt && python -m pytest"

# Parse every file in CORPUS_DIR and print what was extracted (uses OCR in Docker).
report:
	docker compose run --rm --no-deps app python -m app.ingestion.report

# Real startup ingestion, then chunk stats, timings and sample searches.
chunk-report:
	docker compose run --rm --no-deps app python -m app.ingestion.chunk_report

# Retrieve for six sample questions across roles and regions; print doc_ids per
# set (CURRENT / UPCOMING) and the corpus tokens left after the 3,000-token budget.
retrieval-report:
	docker compose run --rm --no-deps app python -m app.retrieval.retrieval_report

# Run each of the five leave tools once on a temporary COPY of the database and
# print the ToolResult of each.
tool-report:
	docker compose run --rm --no-deps app python -m app.tools.tool_report

# Extract holidays from the calendar documents; print counts per calendar and
# the public dates per region (compare them with the source documents).
holiday-report:
	docker compose run --rm --no-deps app python -m app.ingestion.holiday_report

# 15 Part A sample queries against a running service and the real provider.
# Start the stack first (`make up`) and wait for /health 200. Not part of `make test`.
live-smoke:
	python -m app.live_smoke

# Non-Docker setups only: the image downloads the models during `docker build`.
download-models:
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down
