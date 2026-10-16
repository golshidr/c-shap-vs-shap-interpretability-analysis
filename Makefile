# Reproduce everything with `make reproduce` (about 1 hour on 20 CPU cores).
PYTHON ?= python
CONFIG ?= configs/default.yaml
NJOBS  ?= 20

.PHONY: help install data test lint smoke reproduce figures report clean

help:
	@echo "make install    install pinned dependencies + the cshap package"
	@echo "make data       download the OpenML datasets into data/raw"
	@echo "make test       run the unit tests"
	@echo "make lint       run ruff"
	@echo "make smoke      tiny end-to-end run (CI-sized, breast cancer only)"
	@echo "make reproduce  run all experiments, figures and the HTML report"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

smoke:
	$(PYTHON) -m cshap.cli run all --config configs/smoke.yaml

reproduce: data
	OMP_NUM_THREADS=1 $(PYTHON) -m cshap.cli run replicate --config $(CONFIG)
	OMP_NUM_THREADS=1 $(PYTHON) -m cshap.cli run runtime --config $(CONFIG)
	$(PYTHON) -m cshap.cli run sweep --config $(CONFIG) --n-jobs $(NJOBS)
	$(PYTHON) -m cshap.cli run kernel --config $(CONFIG) --n-jobs $(NJOBS)
	$(PYTHON) -m cshap.cli run faithfulness --config $(CONFIG) --n-jobs $(NJOBS)
	$(PYTHON) -m cshap.cli run stability --config $(CONFIG)
	$(MAKE) figures report

figures:
	$(PYTHON) -m cshap.cli figures --config $(CONFIG)

report:
	$(PYTHON) -m cshap.cli report --config $(CONFIG)

clean:
	rm -rf results_smoke .pytest_cache .ruff_cache
