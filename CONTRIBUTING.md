# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[mit license]: https://opensource.org/licenses/MIT
[source code]: https://github.com/phzwart/autocam
[documentation]: https://autocam.readthedocs.io/
[issue tracker]: https://github.com/phzwart/autocam/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

You need Python 3.9+ and the following tools:

- [Conda] (for environment management)
- [Poetry] (for dependency management)
- [Nox] (for testing)

### Step 1: Clone the repository

```console
$ git clone https://github.com/phzwart/autocam.git
$ cd autocam
```

### Step 2: Create and activate the conda environment

```console
$ conda create -n autocam python=3.10 -y
$ conda activate autocam
```

### Step 3: Install Poetry and project dependencies

```console
$ pip install poetry
$ poetry install
```

### Step 4: Install pre-commit hooks (recommended)

```console
$ poetry run pre-commit install
```

This will automatically run formatting and linting checks before each commit.

You can now run an interactive Python session,
or the command-line interface:

```console
$ poetry run python
$ poetry run autocam
```

[conda]: https://docs.conda.io/
[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/

## How to test the project

Make sure you're in the `autocam` conda environment:

```console
$ conda activate autocam
```

Run the full test suite:

```console
$ poetry run nox
```

Or run with a specific Python version:

```console
$ poetry run nox --python=3.10
```

List the available Nox sessions:

```console
$ poetry run nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ poetry run nox --session=tests
```

Or run pre-commit checks:

```console
$ poetry run nox --session=pre-commit
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

**Note**: The project uses pre-commit hooks that automatically format and lint your code before commits. If you haven't installed them, run:

```console
$ poetry run pre-commit install
```

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 85%+ code coverage.
- If your changes add functionality, update the documentation accordingly.
- All pre-commit hooks must pass (black, flake8, isort, etc.).

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ poetry run pre-commit install
```

**Pro tip**: The pre-commit hooks will automatically run when you commit, but you can also run them manually:

```console
$ poetry run pre-commit run --all-files
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/phzwart/autocam/pulls

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md
