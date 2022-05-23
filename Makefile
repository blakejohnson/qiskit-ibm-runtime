.PHONY: lint style black mypy unit-test integration-test staging production

lint:
	pylint -rn programs qka test tools setup.py

style:
	black --check .

black:
	black .

mypy:
	mypy .

unit-test:
	stestr run

integration-test:
	python -m unittest discover -v -s test/integration -t .

staging:
	python -m tools.update $@

production:
	python -m tools.update $@
