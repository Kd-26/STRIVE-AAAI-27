.PHONY: validate test

validate:
	python scripts/validate_notebooks.py
	python scripts/validate_generation_artifact.py data/sample/batch01_generation.zip

test:
	python -m unittest discover -s tests -v
