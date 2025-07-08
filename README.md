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

## Overview

Autocam is a **meta-framework for representation learning** that uses a "conference" metaphor to organize and manage machine learning models. It provides a structured way to define, validate, and run experiments with multiple models across different sessions and working groups.

## Features

- **YAML-driven configuration**: Define conferences, sessions, and participants in simple YAML files
- **Schema validation**: Automatic validation of conference configurations with Pydantic
- **Working group constraints**: Enforce that models in the same working group have compatible parameters
- **Model type restrictions**: Support for CNN, ViT, and custom model types only
- **Direct YAML loading**: No code generation required - load conferences directly from YAML
- **CLI interface**: Command-line tools for validation, info, and running conferences
- **Dynamic training mandates**: Support for different student-target assignment strategies
- **Extensible**: Easy to add new model types and constraints

## Training Mandates

Autocam supports different **training mandates** that define how student-target assignments work:

### 1. **One vs One** (`one_vs_one`)

- One model trains against another specific model
- Fixed student-target pairs defined in the configuration
- Example: CNN trains against ViT, ViT trains against Custom

### 2. **One vs Random Mean** (`one_vs_random_mean`)

- One model trains against the mean of 2+ random other models
- Configurable number of models to use in the mean
- Example: Each model trains against mean of 3 random other models

### 3. **One vs Fixed** (`one_vs_fixed`)

- One model always trains against a fixed target model
- All students train against the same fixed target
- Example: All models train against ViT as fixed target

### 4. **Random Pairs** (`random_pairs`)

- Random student-target pairs each batch
- Completely dynamic assignment
- Example: Random student-target assignment each batch

### 5. **Barycentric Targets** (`barycentric_targets`)

- Random barycentric combinations as targets
- Weighted combinations of multiple models with random weights
- Example: Random barycentric combinations of 2-4 models as targets

## Quick Start

### 1. Create a Conference Configuration

Define your conference in YAML:

```yaml
conference:
  name: "My_Representation_Learning_Conference"
  description: "A conference for my experiments"
  participants:
    - name: "encoder_2d"
      model_type: "cnn"
      model_tag: "encoder"
      in_channels: 3
      out_channels: 64
      dimension: "2D"
      config:
        layers: 3
        kernel_size: 3

    - name: "classifier_2d"
      model_type: "custom"
      model_tag: "classifier"
      in_channels: 64
      out_channels: 10
      dimension: "2D"
      config:
        hidden_layers: [128, 64]

  parallel_sessions:
    - name: "Processing_Pipeline"
      description: "Complete processing pipeline"
      working_groups:
        - name: "Encoder_Group"
          description: "Feature extraction models"
          participants: ["encoder_2d"]
          training_mandate: "one_vs_one"
          student_participants: ["encoder_2d"]
          target_participants: ["classifier_2d"]

        - name: "Classifier_Group"
          description: "Classification models"
          participants: ["classifier_2d"]
          training_mandate: "barycentric_targets"
          student_participants: ["classifier_2d"]
          barycentric_min_models: 2
          barycentric_max_models: 3

metadata:
  version: "1.0.0"
  created_by: "autocam"
```

### 2. Use the Conference in Python

```python
from autocam.conference import create_conference, ModelInterface
from autocam.trainer import DynamicSelfSupervisedTrainer

# Create your model implementations
class MyModel(ModelInterface):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return f"Model {self.name} processed: {x}"

    def get_input_shape(self) -> tuple:
        return self.input_shape

    def get_output_shape(self) -> tuple:
        return self.output_shape

    def train(self):
        # Set training mode
        pass

    def eval(self):
        # Set evaluation mode
        pass

# Load the conference
conference = create_conference("my_conference.yaml")

# Create trainer
models = {
    "encoder_2d": MyModel("encoder_2d", (3, 32, 32), (64, 16, 16)),
    "classifier_2d": MyModel("classifier_2d", (64, 16, 16), (10,))
}

trainer = DynamicSelfSupervisedTrainer(models)

# Train using different mandates
for session in conference.config.parallel_sessions:
    for wg in session.working_groups:
        if wg.training_mandate == "one_vs_one":
            results = trainer.train_batch_one_vs_one(
                batch, wg.student_participants, wg.target_participants,
                loss_fn, optimizers
            )
        elif wg.training_mandate == "barycentric_targets":
            results = trainer.train_batch_barycentric_targets(
                batch, wg.student_participants,
                wg.barycentric_min_models, wg.barycentric_max_models,
                loss_fn, optimizers
            )
```

### 3. Use the CLI

```bash
# Validate a configuration
autocam validate my_conference.yaml

# Show conference info
autocam info my_conference.yaml

# Run a conference
autocam run my_conference.yaml

# List available schemas
autocam list-schemas

# Create an example from a schema
autocam create-example -s schemas/example_conference.yaml
```

## Working Group Constraints

Autocam enforces that all models in a working group must have:

- Same `in_channels`
- Same `out_channels`
- Same `model_tag`

This ensures compatibility within working groups.

## Model Types

Supported model types:

- `cnn`: Convolutional Neural Networks
- `vit`: Vision Transformers
- `custom`: Custom model implementations

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
