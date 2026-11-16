PYTHON ?= python

.PHONY: install quick train-bmm train-idm eval-all

install:
	$(PYTHON) -m pip install -r requirements.txt

quick:
	$(PYTHON) scripts/run_all.py --quick --stage all

train-bmm:
	$(PYTHON) scripts/run_all.py --stage train_bmm

train-idm:
	$(PYTHON) scripts/run_all.py --stage train_idm

eval-all:
	$(PYTHON) scripts/run_all.py --stage eval_all
