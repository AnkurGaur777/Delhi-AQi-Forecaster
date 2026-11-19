.PHONY: install download clean-data tune evaluate slow-models train test lint api dashboard demo-data all

install:
	pip install -r requirements.txt && pip install -e .

download:
	python -m aqi.pipeline download

clean-data:
	python -m aqi.pipeline clean

demo-data:
	python -m aqi.synthetic && python -m aqi.pipeline clean

tune:
	python -m aqi.tune

evaluate:
	python -m aqi.evaluate

slow-models:
	python -m aqi.slow_eval

train:
	python -m aqi.train_final

test:
	pytest -q

lint:
	ruff check src tests app && ruff format --check src tests app

api:
	uvicorn aqi.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/dashboard.py

all: download clean-data tune evaluate slow-models train
