# Autocam

[![PyPI](https://img.shields.io/pypi/v/autocam.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/autocam.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/autocam)][pypi status]
[![License](https://img.shields.io/pypi/l/autocam)][license]

[![Read the documentation at https://autocam.readthedocs.io/](https://img.shields.io/readthedocs/autocam/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/phzwart/autocam/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/phzwart/autocam/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/autocam/
[read the docs]: https://autocam.readthedocs.io/
[tests]: https://github.com/phzwart/autocam/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/phzwart/autocam
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- TODO

## Requirements

- Python 3.9+
- Conda (for environment management)
- Poetry (for dependency management)

## Installation

You can install _Autocam_ via [pip] from [PyPI]:

```console
$ pip install autocam
```

## Development Setup

For development, clone the repository and set up the environment:

```console
$ git clone https://github.com/phzwart/autocam.git
$ cd autocam
$ conda create -n autocam python=3.10 -y
$ conda activate autocam
$ pip install poetry
$ poetry install
$ poetry run pre-commit install
```

Run tests with:

```console
$ poetry run nox
```

## Release Process

**Important: Do NOT create or push version tags manually.**

The release workflow automatically:

- Detects version bumps in `pyproject.toml`
- Creates and pushes version tags (e.g., `v1.2.3`)
- Builds and publishes to PyPI
- Creates GitHub releases with release notes

### To create a new release:

```console
$ python scripts/bump_version.py [major|minor|patch]
$ git add pyproject.toml
$ git commit -m "Bump version to X.Y.Z"
$ git push
```

**Do NOT run `git tag` or `git push --tags`.**

See [Contributing] for more details.

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Autocam_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/phzwart/autocam/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/phzwart/autocam/blob/main/LICENSE
[contributor guide]: https://github.com/phzwart/autocam/blob/main/CONTRIBUTING.md
[command-line reference]: https://autocam.readthedocs.io/en/latest/usage.html
