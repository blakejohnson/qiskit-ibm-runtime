.PHONY: test

test:
	python -m unittest discover -v -s test -t .
