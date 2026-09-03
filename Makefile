.PHONY: install test lint type doctor census plan

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

type:
	mypy src

doctor:
	model-profiler doctor
