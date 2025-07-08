#!/usr/bin/env python3
"""Example demonstrating different training mandates in Autocam."""

import numpy as np

from autocam.conference import ModelInterface
from autocam.conference import create_conference
from autocam.trainer import DynamicSelfSupervisedTrainer


class DummyModel(ModelInterface):
    """A dummy model for demonstration purposes."""

    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        """Initialize DummyModel with name, input shape, and output shape."""
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(*output_shape)
        self.training_mode = True

    def forward(self, x):
        """Forward pass - returns a random numpy array of the correct shape."""
        # Simulate a numeric output for compatibility with trainer
        batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
        # Output shape: (batch_size, ...) if output is not scalar
        if isinstance(self.output_shape, tuple) and len(self.output_shape) > 0:
            return np.random.randn(batch_size, *self.output_shape)
        else:
            return np.random.randn(batch_size)

    def get_input_shape(self) -> tuple:
        """Return the input shape of the model."""
        return self.input_shape

    def get_output_shape(self) -> tuple:
        """Return the output shape of the model."""
        return self.output_shape

    def train(self):
        """Set the model to training mode."""
        self.training_mode = True

    def eval(self):
        """Set the model to evaluation mode."""
        self.training_mode = False


class DummyLoss:
    """A dummy loss function for demonstration."""

    def __call__(self, student_output, target_output):
        # Simulate loss computation
        return DummyLossValue(0.5)


class DummyLossValue:
    """A dummy loss value that simulates PyTorch tensor behavior."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def backward(self):
        pass


class DummyOptimizer:
    """A dummy optimizer for demonstration."""

    def __init__(self, model_name):
        """Initialize the dummy optimizer."""
        self.model_name = model_name

    def zero_grad(self):
        """Zero gradients (dummy implementation)."""
        pass

    def step(self):
        """Step optimizer (dummy implementation)."""
        pass


def create_dummy_models():
    """Create dummy model instances for demonstration."""
    models = {}

    models["model_cnn_2d"] = DummyModel("model_cnn_2d", (3, 32, 32), (128,))
    models["model_vit_2d"] = DummyModel("model_vit_2d", (3, 32, 32), (128,))
    models["model_custom_2d"] = DummyModel("model_custom_2d", (3, 32, 32), (128,))
    models["model_mlp_2d"] = DummyModel("model_mlp_2d", (3, 32, 32), (128,))

    return models


def create_dummy_optimizers(model_names):
    """Create dummy optimizers for each model."""
    return {name: DummyOptimizer(name) for name in model_names}


def demonstrate_training_mandates():
    """Demonstrate different training mandates with reduced complexity."""
    print("🎓 Training Mandates Demonstration")
    print("=" * 50)

    # Load the conference
    conference = create_conference("schemas/training_mandates_conference.yaml")

    # Create models and trainer
    models = create_dummy_models()
    trainer = DynamicSelfSupervisedTrainer(models)
    optimizers = create_dummy_optimizers(models.keys())
    loss_fn = DummyLoss()

    # Create dummy batch
    dummy_batch = np.random.randn(16, 3, 32, 32)  # Batch of 16 images

    print(f"📊 Conference: {conference.name}")
    print(f"📋 Sessions: {conference.list_sessions()}")
    print(f"👥 Models: {list(models.keys())}")
    print()

    def handle_mandate(wg, session_name):
        """Handle a single working group's training mandate."""
        print(f"Working Group: {wg.name}")
        print(f"Training Mandate: {wg.training_mandate}")
        print(f"Participants: {wg.participants}")
        if wg.training_mandate == "one_vs_one":
            results = trainer.train_batch_one_vs_one(
                dummy_batch,
                wg.student_participants or [],
                wg.target_participants or [],
                loss_fn,
                optimizers,
            )
            print(f"  Results: {list(results.keys())}")
        elif wg.training_mandate == "one_vs_random_mean":
            results = trainer.train_batch_one_vs_random_mean(
                dummy_batch,
                wg.student_participants or [],
                wg.random_mean_count or 2,
                loss_fn,
                optimizers,
            )
            print(f"  Results: {list(results.keys())}")
        elif wg.training_mandate == "one_vs_fixed":
            if wg.fixed_target:
                results = trainer.train_batch_one_vs_fixed(
                    dummy_batch,
                    wg.student_participants or [],
                    wg.fixed_target,
                    loss_fn,
                    optimizers,
                )
                print(f"  Results: {list(results.keys())}")
        elif wg.training_mandate == "random_pairs":
            results = trainer.train_batch_random_pairs(
                dummy_batch, wg.participants, loss_fn, optimizers
            )
            print(f"  Results: {list(results.keys())}")
        elif wg.training_mandate == "barycentric_targets":
            results = trainer.train_batch_barycentric_targets(
                dummy_batch,
                wg.student_participants or [],
                wg.barycentric_min_models or 2,
                wg.barycentric_max_models or 3,
                loss_fn,
                optimizers,
            )
            print(f"  Results: {list(results.keys())}")
        print(f"  Training epochs: {wg.training_epochs}")
        print()

    # Demonstrate each training mandate
    for session_name in conference.list_sessions():
        print(f"📚 Session: {session_name}")
        print("-" * 30)
        session = conference.get_session(session_name)
        if not session:
            continue
        for wg in session.working_groups:
            handle_mandate(wg, session_name)

    # Demonstrate validation
    print("🔍 Validation Example")
    print("-" * 20)
    dummy_dataloader = [np.random.randn(8, 3, 32, 32) for _ in range(3)]
    validation_results = trainer.validate_linear_reconstruction(
        dummy_dataloader, list(models.keys())
    )
    print("Linear reconstruction scores:")
    for model_name, score in validation_results.items():
        print(f"  {model_name}: {score:.4f}")
    print("\n✅ Training mandates demonstration completed!")


if __name__ == "__main__":
    demonstrate_training_mandates()
