.PHONY: lint style black mypy unit-test integration-test staging production

lint:
	pylint -rn programs qka test tools

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

deploy-staging:
	python -m tools.update $@

deploy-production:
	python -m tools.update $@
