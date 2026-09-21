# Cardessa Ammonix factory targets. Milestones add targets as they land;
# every target's checker prints a one-line JSON verdict (transcript-evidence rule).

.PHONY: test lint check-m0 determinism toy check-toy world check-world

test:
	cd ammonix_core && python -m pytest -q
	python -m pytest cardessa/tests -q

lint:
	cd ammonix_core && python -m ruff check .
	python -m ruff check --config ammonix_core/pyproject.toml cardessa scripts

check-m0:
	python scripts/check_m0.py

determinism:
	python scripts/determinism_check.py

toy:
	python scripts/build_toy.py

check-toy:
	python scripts/check_toy.py

world:
	python scripts/build_world.py

check-world:
	python scripts/check_world.py

corpus:
	python scripts/build_corpus.py

check-corpus:
	python scripts/check_corpus.py
