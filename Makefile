# Convenience targets for PADS development.
.PHONY: install dev test web gradio tracking bench clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev,viz,web,gradio]"

test:
	pytest tests/ -v

web:
	streamlit run web/app.py

gradio:
	python -m pads.gradio_app

tracking:
	python -m pads.tracking_demo

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
