.PHONY: install data train test lint serve docker clean

install:
	pip install -r requirements-dev.txt && pip install -e .

data:
	python -m churn_ml.cli generate-data

train:
	python -m churn_ml.cli train

test:
	pytest

lint:
	ruff check src tests

serve:
	uvicorn churn_ml.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t churn-ml:latest . && docker run --rm -p 8000:8000 churn-ml:latest

clean:
	rm -rf artifacts data .pytest_cache .ruff_cache
