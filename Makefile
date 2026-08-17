install:
	uv sync

run: install
	uv run python3 -m src

debug: install
	uv run python3 -m pdb -m src 

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONEY: install run debug clean lint lint-strict