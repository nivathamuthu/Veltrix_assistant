.PHONY: install test test-docker report chunk-report download-models run up down

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

# Non-Docker setups only: the image downloads the models during `docker build`.
download-models:
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down
