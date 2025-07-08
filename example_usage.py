#!/usr/bin/env python3
"""Example usage of the conference system without code generation."""

from autocam.conference import ModelInterface
from autocam.conference import create_conference


class SimpleModel(ModelInterface):
    """A simple model implementation."""

    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        """Initialize the simple model."""
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        """Forward pass - just return the input for demo."""
        return f"Model {self.name} processed: {x}"

    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        return self.input_shape

    def get_output_shape(self) -> tuple:
        """Get the output shape."""
        return self.output_shape


def main():
    """Example of using the conference system directly."""
    # Create conference from YAML file
    conference = create_conference("schemas/example_conference.yaml")

    print(f"Conference: {conference.name}")
    print(f"Sessions: {conference.list_sessions()}")
    print(f"Participants: {conference.list_participants()}")

    # Create some mock models
    models = {
        "encoder_2d": SimpleModel("encoder_2d", (3, 32, 32), (64, 16, 16)),
        "encoder_3d": SimpleModel("encoder_3d", (1, 32, 32, 32), (32, 16, 16, 16)),
        "classifier_2d": SimpleModel("classifier_2d", (64, 16, 16), (10,)),
        "classifier_3d": SimpleModel("classifier_3d", (32, 16, 16, 16), (5,)),
        "transformer_2d": SimpleModel("transformer_2d", (64, 16, 16), (64, 16, 16)),
        "transformer_3d": SimpleModel(
            "transformer_3d", (32, 16, 16, 16), (32, 16, 16, 16)
        ),
    }

    # Register models with the conference
    for name, model in models.items():
        participant_config = conference.get_participant_config(name)
        if participant_config:
            spec = {
                "name": participant_config.name,
                "model_type": participant_config.model_type.value,
                "model_tag": participant_config.model_tag,
                "in_channels": participant_config.in_channels,
                "out_channels": participant_config.out_channels,
                "dimension": participant_config.dimension.value,
                "config": participant_config.config or {},
            }
            conference.register_participant(name, model, spec)
            print(f"Registered {name}")

    # Validate that all participants are registered
    if conference.validate_registration():
        print("✓ All participants registered successfully")
    else:
        print("✗ Some participants missing")

    # Run a session
    print("\nRunning session 'Processing_Pipelines':")
    results = conference.run_session("Processing_Pipelines", "test_data")
    for model_name, result in results.items():
        print(f"  {model_name}: {result}")

    # Get participants by tag
    print(f"\nEncoders: {conference.get_participants_by_tag('encoder')}")
    print(f"Classifiers: {conference.get_participants_by_tag('classifier')}")
    print(f"Attention models: {conference.get_participants_by_tag('attention')}")


if __name__ == "__main__":
    main()
