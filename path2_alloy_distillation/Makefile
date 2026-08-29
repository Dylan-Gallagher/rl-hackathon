.PHONY: test lint format-check compile smoke

test:
	python -m pytest

lint:
	ruff check .

format-check:
	ruff format --check .

compile:
	python -m compileall -q src tests

smoke:
	bash scripts/smoke.sh
