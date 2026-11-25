PYTHON := .venv/Scripts/python
PYTEST := .venv/Scripts/pytest

.PHONY: test test-m5 demo-m5 reports-m4 serve

test:
	PYTHONPATH=. $(PYTEST) tests_v2 -v

test-m5:
	PYTHONPATH=. $(PYTEST) tests_v2/test_m5_serving.py -v

demo-m5:
	$(PYTHON) scripts/demo_m5.py --lam auto:balanced --prompts 50

reports-m4:
	PYTHONPATH=. $(PYTHON) -m eval.m4_router_evaluation
	PYTHONPATH=. $(PYTHON) -m eval.m4_calibration
	PYTHONPATH=. $(PYTHON) -m eval.m4_ood_evaluation
	PYTHONPATH=. $(PYTHON) -m eval.m4_routellm_baseline

serve:
	.venv/Scripts/uvicorn main:app --reload --port 8000
