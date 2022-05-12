.PHONY: lint style black test staging production

lint:
	pylint -rn programs qka test tools

style:
	black --check .

black:
	black .

test:
	python -m unittest discover -v -s test -t .

staging:
	python -m tools.update $@

production:
	python -m tools.update $@
