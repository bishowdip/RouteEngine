.PHONY: install test figures gui clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# Single command to regenerate every figure/table into figures/
figures:
	python -m src.bench.run_all

# Launch the interactive desktop GUI
gui:
	python -m src.gui

clean:
	rm -rf figures/*.png figures/*.html
	find . -name __pycache__ -type d -exec rm -rf {} +
