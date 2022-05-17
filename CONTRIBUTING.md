Contributing
============

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
3. If it makes sense for your change that you have added new tests that
   cover the changes.

### Style guide

Please submit clean code and please make effort to follow existing
conventions in order to keep it as readable as possible. We use:
* [Pylint](https://www.pylint.org) linter
* [Black](https://pypi.org/project/black/) style
* [mypy](http://mypy-lang.org/) type hinting

To ensure your changes respect the style guidelines, you can run the following
commands:

All platforms:

``` {.bash}
$ make lint
$ make style
$ make mypy
```

You can run below command to autofix style issues reported when running `make style`.
``` {.bash}
$ make black
```

### Test

#### Test Types
There are two different types of tests in `ntc-ibm-programs`. The implementation is based upon the well-documented [unittest](https://docs.python.org/3/library/unittest.html) Unit testing framework.

##### 1. Unit tests
Run locally without connecting to an external system. They are short-running, stable and give a basic level of confidence during development.

To execute all unit tests, run:
``` {.bash}
$ make unit-test
```
##### 2. Integration tests
Executed against an external system configured via a (token, instance, url) tuple. Detailed coverage of happy and non-happy paths. They are long-running and unstable at times. A successful test run gives a high level of confidence that programs work well.

To execute all integration tests, run:
``` {.bash}
$ make integration-test
```
