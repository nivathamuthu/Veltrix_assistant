.PHONY: install test run up down

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down
