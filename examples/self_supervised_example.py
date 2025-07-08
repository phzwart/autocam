#!/usr/bin/env python3
"""Example demonstrating self-supervised learning with dynamic role switching."""

import numpy as np

from autocam.conference import ModelInterface
from autocam.conference import create_conference


class SelfSupervisedModel(ModelInterface):
    """A model that can act as both student and target in self-supervised learning."""

    def __init__(
        self, name: str, input_shape: tuple, output_shape: tuple, model_type: str
    ):
        """Initialize the self-supervised model."""
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_type = model_type
        self.current_role = None  # Will be set dynamically by the conference
        self.epochs_trained = 0

        # Initialize random weights (in real implementation, this would be actual model weights)
        self.weights = np.random.randn(*output_shape)

    def forward(self, x):
        """Forward pass - simulates model inference."""
        # In real implementation, this would be actual model forward pass
        return (
            f"{self.name} ({self.current_role}) processed: {x.shape} -> "
            f"{self.output_shape}"
        )

    def get_input_shape(self) -> tuple:
        """Get the input shape of the model."""
        return self.input_shape

    def get_output_shape(self) -> tuple:
        """Get the output shape of the model."""
        return self.output_shape

    def set_role(self, role: str):
        """Set the current role (student or target) for this session."""
        self.current_role = role

    def train_epoch(self, target_outputs=None):
        """Train for one epoch using target outputs as supervision."""
        if self.current_role == "student" and target_outputs is not None:
            # Simulate training update (in real implementation, this would be actual training)
            self.weights += 0.01 * np.random.randn(*self.weights.shape)
            self.epochs_trained += 1
            return f"Trained {self.name} (student) for epoch {self.epochs_trained}"
        elif self.current_role == "target":
            return f"{self.name} acting as target - no training"
        return f"No training for {self.name} (role: {self.current_role})"

    def get_representation(self):
        """Get the current representation (weights)."""
        return self.weights.copy()


def create_self_supervised_models():
    """Create model instances for the self-supervised learning example."""
    models = {}

    # Create models that can switch between student and target roles
    models["model_cnn_2d"] = SelfSupervisedModel(
        "model_cnn_2d",
        (3, 32, 32),
        (128,),
        "cnn",
    )
    models["model_vit_2d"] = SelfSupervisedModel(
        "model_vit_2d",
        (3, 32, 32),
        (128,),
        "vit",
    )
    models["model_custom_2d"] = SelfSupervisedModel(
        "model_custom_2d",
        (3, 32, 32),
        (128,),
        "custom",
    )

    return models


def run_self_supervised_conference():
    """Run the self-supervised learning conference with dynamic role switching."""
    print("🎓 Starting Self-Supervised Representation Learning Conference")
    print("🔄 Dynamic Role Switching: Every model can be student and target")
    print("=" * 70)

    # Load the conference
    conference = create_conference("schemas/self_supervised_conference.yaml")

    # Create and register models
    models = create_self_supervised_models()
    for name, model in models.items():
        conference.register_participant(name, model, {})

    # Run each session with dynamic role assignment
    sessions = [
        "Session_1_Cross_Architecture_Learning",
        "Session_2_Cyclic_Learning",
        "Session_3_Ensemble_Learning",
        "Session_4_Final_Convergence",
    ]

    for session_name in sessions:
        print(f"\n📚 Running Session: {session_name}")
        print("-" * 50)

        # Get session info
        session = conference.get_session(session_name)
        if session:
            print(f"Training epochs: {session.training_epochs}")

            # Set roles for this session based on working group configuration
            for wg in session.working_groups:
                print(f"\nWorking Group: {wg.name}")
                print(f"Description: {wg.description}")

                # Set student roles
                if wg.student_participants:
                    for student_name in wg.student_participants:
                        if student_name in models:
                            models[student_name].set_role("student")
                            print(f"  {student_name} -> STUDENT")

                # Set target roles
                if wg.target_participants:
                    for target_name in wg.target_participants:
                        if target_name in models:
                            models[target_name].set_role("target")
                            print(f"  {target_name} -> TARGET")

            # Simulate training for this session
            print(f"\n🏋️ Training for {session.training_epochs} epochs...")
            for epoch in range(min(5, session.training_epochs)):  # Show first 5 epochs
                for _name, model in models.items():
                    if model.current_role == "student":
                        result = model.train_epoch("target_outputs")
                        print(f"  Epoch {epoch+1}: {result}")

            # Show model states after session
            print("\n📊 Model states after session:")
            for name, model in models.items():
                print(f"  {name}: {model.epochs_trained} epochs trained")

    # Final convergence: collect all representations
    print("\n🎯 Final Convergence - Collecting Representations")
    print("-" * 50)

    representations = []
    for name, model in models.items():
        rep = model.get_representation()
        representations.append(rep)
        print(f"  {name}: representation shape {rep.shape}")

    # Stack representations and perform SVD
    if representations:
        stacked_reps = np.stack(representations, axis=0)
        print(f"\n📊 Stacked representations shape: {stacked_reps.shape}")

        # Perform SVD (simulated)
        print("🔍 Performing SVD on stacked representations...")
        # In real implementation: U, S, Vt = np.linalg.svd(stacked_reps)
        print("✅ SVD completed - final latent vectors extracted!")

        print("🎉 Self-supervised learning conference completed!")
        total_epochs = sum(m.epochs_trained for m in models.values())
        print(f"📈 Total training epochs across all models: {total_epochs}")


if __name__ == "__main__":
    run_self_supervised_conference()
