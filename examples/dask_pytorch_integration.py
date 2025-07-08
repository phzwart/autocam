"""Example: Dask + PyTorch integration for Autocam conference execution."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dask.distributed import Client
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from autocam.conference import Conference
from autocam.orchestrator import AutocamDaskOrchestrator


class DummyModel(nn.Module):
    """Dummy model for demonstration."""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """Initialize dummy model."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64), nn.ReLU(), nn.Linear(64, out_channels)
        )

    def forward(self, x):
        """Forward pass."""
        # Ensure input is 2D for linear layers
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


def create_dummy_dataloader(batch_size: int = 32, num_batches: int = 10):
    """Create dummy data for demonstration."""
    # Create dummy data
    data = torch.randn(num_batches * batch_size, 3, 32, 32)  # 3-channel images
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_optimizer_factory(learning_rate: float = 0.001):
    """Create optimizer factory function."""
    def optimizer_factory(model):
        return optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer_factory


def create_loss_function():
    """Create loss function for self-supervised learning."""
    def loss_fn(student_output, target_output):
        # Simple MSE loss between student and target outputs
        return nn.MSELoss()(student_output, target_output)

    return loss_fn


def create_dask_client():
    """Create and configure a Dask client for distributed computing."""
    client = Client(n_workers=2, threads_per_worker=2)
    return client


def main():
    """Run the Dask + PyTorch integration example."""
    # Load conference configuration
    config_path = Path("schemas/training_mandates_conference.yaml")
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    conference = Conference.from_dict(config_data)

    # Create model implementations
    model_implementations = {}
    for participant in conference.config.participants:
        model = DummyModel(
            in_channels=participant.in_channels,
            out_channels=participant.out_channels,
            **participant.config,
        )
        model_implementations[participant.name] = model

    # Create data and training components
    dataloader = create_dummy_dataloader()
    loss_fn = create_loss_function()
    optimizer_factory = create_optimizer_factory()

    # Initialize orchestrator
    orchestrator = AutocamDaskOrchestrator(
        n_workers=2, gpus_per_worker=1, gpu_memory_limit="4GB"  # Use 2 workers for demo
    )

    try:
        print("Starting conference execution with Dask + PyTorch...")
        print(f"Conference: {conference.config.name}")
        print(f"Participants: {[p.name for p in conference.config.participants]}")
        print(f"Parallel Sessions: {len(conference.config.parallel_sessions)}")

        # Run the conference
        results = orchestrator.run_conference(
            conference=conference,
            model_implementations=model_implementations,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer_factory=optimizer_factory,
        )

        print("\nConference execution completed!")
        print(f"Results from {len(results)} working groups:")

        for result in results:
            print(f"\nWorking Group: {result['working_group']}")
            print(f"Participants: {result['participants']}")
            print("Training completed successfully")

    except Exception as e:
        print(f"Error during conference execution: {e}")
        raise
    finally:
        # Clean up
        orchestrator.close()
        print("\nOrchestrator closed.")


if __name__ == "__main__":
    main()
