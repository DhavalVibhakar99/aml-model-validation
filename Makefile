PY := .venv/bin/python

.PHONY: setup data split train validate all test holdout

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) src/download.py
	$(PY) src/ingest.py --dataset HI-Small
	$(PY) src/ingest.py --dataset LI-Small

split:
	$(PY) src/split.py

train:
	$(PY) src/train.py

validate:
	$(PY) src/validate.py

all: split train validate

test:
	$(PY) -m pytest -q

# scores the final holdout ONCE; refuses to rerun (DECISIONS #11)
holdout:
	$(PY) src/holdout.py
