# Convenience targets for PADS development.
.PHONY: install dev test web gradio bench clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev,viz,web,gradio]"

test:
	pytest tests/ -v

web:
	streamlit run web/app.py

gradio:
	gradio web/gradio_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
