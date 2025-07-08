# Autocam Conference System

A meta-framework for organizing representation learning experiments through a "conference" metaphor. This system allows you to define participants (models), organize them into working groups, and run parallel sessions without creating the actual neural network models.

## Overview

The system uses a conference metaphor where:

- **Participants**: Models with specifications (name, type, dimensions, etc.)
- **Working Groups**: Collections of participants that work together
- **Parallel Sessions**: Multiple working groups that run simultaneously
- **Conference**: The overall organization containing all sessions and participants

## Key Features

- **YAML Configuration**: Define your conference structure in YAML
- **Type Safety**: Full Pydantic validation with type hints
- **Code Generation**: Automatically generate Python code from YAML
- **Model Injection**: Register your actual models later (no model creation here)
- **Validation**: Ensure participants only appear once per session

## Installation

```bash
# Activate conda environment
conda activate autocam

# Install dependencies
poetry install
```

## Quick Start

### 1. Create an Example Configuration

```bash
autocam create-example
```

This creates `example_conference.yaml` with a sample configuration.

### 2. Generate Python Code

```bash
autocam generate example_conference.yaml
```

This generates a Python package with the conference structure.

### 3. Use the Generated Code

```python
# Import the generated conference
from generated_conference.conference import conference, ModelInterface

# Create your actual model (this is where you inject your models)
class MyEncoder(ModelInterface):
    def forward(self, x):
        # Your actual model implementation
        return processed_x

    def get_input_shape(self):
        return (3, 32, 32)

    def get_output_shape(self):
        return (64,)

# Register your model
my_model = MyEncoder()
conference.register_participant("encoder_2d", my_model, {
    "name": "encoder_2d",
    "model_type": "cnn",
    "model_tag": "encoder",
    "in_channels": 3,
    "out_channels": 64,
    "dimension": "2D"
})

# Run a session
results = conference.run_session("Processing_Pipelines", your_data)
```

## YAML Configuration Structure

```yaml
conference:
  name: "My_Conference"
  description: "A conference for representation learning"
  participants:
    - name: "encoder_2d"
      model_type: "cnn"
      model_tag: "encoder"
      in_channels: 3
      out_channels: 64
      dimension: "2D"
      model_config:
        layers: 3
        kernel_size: 3

    - name: "classifier_2d"
      model_type: "mlp"
      model_tag: "classifier"
      in_channels: 64
      out_channels: 10
      dimension: "2D"
      model_config:
        hidden_layers: [128, 64]

  parallel_sessions:
    - name: "Processing_Pipelines"
      description: "Complete processing pipelines"
      working_groups:
        - name: "2D_Pipeline"
          description: "2D image processing"
          participants: ["encoder_2d", "classifier_2d"]

    - name: "Model_Comparison"
      description: "Compare similar models"
      working_groups:
        - name: "Encoder_Group"
          description: "All encoders"
          participants: ["encoder_2d"]

metadata:
  version: "1.0.0"
  created_by: "autocam"
```

## CLI Commands

```bash
# Create example configuration
autocam create-example

# Validate a YAML file
autocam validate my_conference.yaml

# Generate Python code from YAML
autocam generate my_conference.yaml

# Generate example files
autocam generate-example

# Show information about a configuration
autocam info my_conference.yaml
```

## Model Types

Supported model types:

- `cnn`: Convolutional Neural Networks
- `resnet`: Residual Networks
- `vit`: Vision Transformers
- `transformer`: Transformer models
- `mlp`: Multi-Layer Perceptrons
- `custom`: Custom models

## Dimensions

- `2D`: 2D models (images)
- `3D`: 3D models (volumes)

## Model Interface

When you inject your models, they should implement:

```python
class ModelInterface(ABC):
    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get the output shape."""
        pass
```

## Working with Sessions

```python
# Get all participants in a session
participants = conference.get_session_participants("Processing_Pipelines")

# Get a specific working group
wg = conference.get_working_group("Processing_Pipelines", "2D_Pipeline")

# Get participants by tag
encoders = conference.get_participants_by_tag("encoder")

# Validate that all expected participants are registered
is_valid = conference.validate_registration()
```

## Example Use Cases

### 1. Multi-Modal Learning

Organize models for different modalities (image, text, audio) into working groups.

### 2. Model Comparison

Compare different architectures for the same task in parallel sessions.

### 3. Pipeline Testing

Test complete processing pipelines with different model combinations.

### 4. Hyperparameter Studies

Organize models with different hyperparameters into working groups.

## Validation Rules

1. Each participant name must be unique
2. Participants in working groups must exist in the conference
3. Each participant can only appear once per parallel session
4. Working groups must have at least one participant
5. Parallel sessions must have at least one working group

## Contributing

This is a meta-framework - the actual neural network models are injected by you. The system provides:

- Organization structure
- Validation
- Code generation
- Session management

You provide:

- Actual model implementations
- Training loops
- Data processing
- Evaluation metrics

## License

MIT License - see LICENSE file for details.
