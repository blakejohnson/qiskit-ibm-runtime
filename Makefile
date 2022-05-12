.PHONY: lint test staging production

lint:
	pylint -rn programs qka test tools

test:
	python -m unittest discover -v -s test -t .

staging:
	python -m tools.update $@

production:
	python -m tools.update $@
