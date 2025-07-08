# Conference YAML Schemas

This directory contains YAML schema files and examples for the Autocam Conference System.

## Files

### `conference_schema.yaml`

The main schema definition file that documents the structure and requirements for conference configurations.

### `participant_schema.yaml`

Detailed schema for participant definitions, including model types, dimensions, and configuration examples.

### `session_schema.yaml`

Detailed schema for parallel sessions and working groups, including session patterns and validation rules.

### `example_conference.yaml`

A comprehensive example showing a full conference with multiple participants, sessions, and working groups.

### `minimal_conference.yaml`

A minimal example showing the simplest possible conference configuration.

## Schema Structure

The schema is split into multiple files for better organization:

### Conference (`conference_schema.yaml`)

- `name`: String - Name of the conference
- `description`: String (optional) - Description of the conference
- `participants`: List of participant objects (see `participant_schema.yaml`)
- `parallel_sessions`: List of session objects (see `session_schema.yaml`)

### Participant (`participant_schema.yaml`)

- `name`: String - Unique identifier for the participant
- `model_type`: String - Type of neural network model (**must be one of:** `cnn`, `vit`, `custom`)
- `model_tag`: String - Tag for grouping models (**must match within a working group**)
- `in_channels`: Integer - Number of input channels (**must match within a working group**)
- `out_channels`: Integer - Number of output channels (**must match within a working group**)
- `dimension`: String - 2D or 3D model (`2D`, `3D`)
- `config`: Object (optional) - Additional model configuration parameters

### Session (`session_schema.yaml`)

- `name`: String - Name of the parallel session
- `description`: String (optional) - Description of the parallel session
- `working_groups`: List of working group objects

### Working Group (`session_schema.yaml`)

- `name`: String - Name of the working group
- `description`: String (optional) - Description of the working group
- `participants`: List of strings - Participant names in this group

## Model Type Constraints

- Only the following model types are allowed: `cnn`, `vit`, `custom`

## Working Group Constraints

- **All models in a working group must have:**
  - The same number of input channels (`in_channels`)
  - The same number of output channels (`out_channels`)
  - The same `model_tag`

## Model Configuration Examples

### CNN Configuration

```yaml
config:
  layers: 3
  kernel_size: 3
```

### ViT Configuration

```yaml
config:
  patch_size: 16
  num_heads: 8
  num_layers: 6
  hidden_size: 512
```

### Custom Model Configuration

```yaml
config:
  custom_param: value
```

## Validation Rules

1. Each participant name must be unique
2. Participants in working groups must exist in the conference
3. Each participant can only appear once per parallel session
4. Working groups must have at least one participant
5. Parallel sessions must have at least one working group
6. **All models in a working group must have the same `in_channels`, `out_channels`, and `model_tag`**
7. `model_type` must be one of: `cnn`, `vit`, `custom`

## Usage

```bash
# List all available schemas
autocam list-schemas

# Create example configuration
autocam create-example -s schemas/minimal_conference.yaml

# Validate a YAML file
autocam validate schemas/example_conference.yaml

# Generate Python code from YAML
autocam generate schemas/example_conference.yaml

# Show information about a configuration
autocam info schemas/example_conference.yaml
```
