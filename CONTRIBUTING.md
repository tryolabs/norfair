# Contributing to Norfair

Thank you for your interest in contributing to Norfair! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it:
    - Post your intended feature in an issue, and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug fix for an outstanding issue.
    - Search for your issue in the [list](https://github.com/tryolabs/norfair/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide it.

Once you implement and test your feature or bug fix, please submit a Pull Request to https://github.com/tryolabs/norfair/pulls.


# Setup

1. Clone this repository `git clone git@github.com:tryolabs/norfair.git`.
2. Set up Python 3.10+ (we test on 3.10, 3.11, 3.12, 3.13). Using [pyenv](https://github.com/pyenv/pyenv) is highly recommended.
3. Install [poetry](https://python-poetry.org/docs/#installation) version 1.2 or above.
4. Install dependencies `poetry install --all-extras`.

In the following commands, we will include `poetry run <cmd>` when a command needs the virtual environment. This is not necessary if you activate it by running `poetry shell` once.

## Formatting

We use [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/) to automatically format our python code. It's recommended that you configure them on your editor of choice. If you don't, you'll likely get a linting error on the PR

For VSCode, follow the setup recommended [here](https://code.visualstudio.com/docs/python/editing).

Alternatively, make sure to run `poetry run black .` and `poetry run isort .` on the root directory before committing.

## Running tests locally

The tests are automatically checked on each PR by a GitHub Action. For this reason, you are encouraged to skip this setup and send a PR without testing it locally first. Delaying this step until the tests fail on the GitHub Action if they ever do.

Tests are run with tox using `poetry run tox`.

You will likely receive an error where tox is not able to find the python versions necessary, to solve this with pyenv:

1. List installed versions with `pyenv versions`.
2. Make sure you have at least one version installed for each Python `3.10`, `3.11`, `3.12`, and `3.13`. Available versions can be found with `pyenv install --list` and installed with `pyenv install X.X.X`.
3. Once you have one version of each Python run `pyenv local 3.10.X 3.11.X 3.12.X 3.13.X`. Substitute `X` with the specific versions listed in `pyenv versions`.

Tox will run the unit tests on all supported Python versions and one integration test that checks the performance in the MOT Challenge. This integration test can take a few minutes and needs an internet connection. The MOT metrics environment pins NumPy < 2 automatically to remain compatible with `motmetrics`.

## Documentation

Any suggestion on how to improve the documentation is welcome and don't feel obligated to set up the repo locally to contribute. Simply create an issue describing the change and we will take care of it.

Nevertheless, if you still want to test the change first and create the PR yourself, follow these steps:

1. Install documentation dependencies `poetry run pip install -r docs/requirements.txt`.
2. Start the debugging server `poetry run mkdocs serve` and open http://localhost:8000.
3. The above version is useful for debugging but it doesn't include the versioning. Once you are happy with the result you can see the final result with run `poetry run mike deploy dev` and `poetry run mike serve`. Open the browser and switch to `dev` version.

Our documentation follows [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) docstring format.
