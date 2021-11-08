.PHONY: test update publish cleanup

test:
	python -m unittest discover -v -s test -t .

update:
	python -m tools.ci_script update

publish:
	python -m tools.ci_script publish

cleanup:
	python -m tools.ci_script cleanup