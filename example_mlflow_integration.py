#!/usr/bin/env python3
"""Example demonstrating MLflow integration with Autocam distributed training."""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from dask.distributed import Client, LocalCluster
import numpy as np

from autocam.models import Participant, ModelType, Dimension, TrainingMandate
from autocam.conference_builder import create_conference_grid
from autocam.distributed_trainer import DistributedMandateTrainer, GPUAllocator
from autocam.mlflow_tracker import AutocamMLflowTracker


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, name: str, in_channels: int, out_channels: int):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x
    
    def get_input_shape(self):
        return (3, 32, 32)
    
    def get_output_shape(self):
        return (self.conv.out_channels,)
    
    def train(self):
        super().train()
    
    def eval(self):
        super().eval()


def create_models(participants):
    """Create model implementations from participants."""
    models = {}
    for participant in participants:
        model = SimpleModel(
            participant.name,
            participant.in_channels,
            participant.out_channels
        )
        models[participant.name] = model
    return models


def create_optimizer_factory(learning_rate: float = 0.001):
    """Create optimizer factory function."""
    def optimizer_factory(model):
        return optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer_factory


def create_dummy_dataloader(batch_size: int = 16, num_batches: int = 10):
    """Create dummy dataloader for demonstration."""
    for _ in range(num_batches):
        yield torch.randn(batch_size, 3, 32, 32)


def loss_function(student_output, target_output):
    """Simple MSE loss function."""
    return nn.MSELoss()(student_output, target_output)


def demonstrate_mlflow_tracking():
    """Demonstrate MLflow tracking with different mandates."""
    print("🔍 MLflow Tracking Demo")
    print("=" * 50)
    
    # Create MLflow tracker
    mlflow_tracker = AutocamMLflowTracker(
        experiment_name="autocam_demo",
        log_artifacts=True,
        log_models=True
    )
    
    # Create participants
    participants = [
        Participant(
            name=f"model_{i}",
            model_type=ModelType.CNN,
            model_tag="encoder",
            in_channels=3,
            out_channels=128,
            dimension=Dimension.D2,
            config={"layers": 3, "kernel_size": 3}
        )
        for i in range(4)
    ]
    
    # Create conference with different mandates
    conference_config = create_conference_grid(
        participants=participants,
        num_sessions=2,
        working_groups_per_session=1,
        training_mandate=TrainingMandate.ONE_VS_ONE,
        mixing_strategy="max"
    )
    
    # Create models and trainer
    models = create_models(participants)
    gpu_allocator = GPUAllocator(["gpu:0"])  # Single GPU for demo
    trainer = DistributedMandateTrainer(gpu_allocator, mlflow_tracker)
    
    # Create dataloader and optimizer factory
    dataloader = create_dummy_dataloader()
    optimizer_factory = create_optimizer_factory()
    
    # Train conference with MLflow tracking
    results = trainer.train_conference_distributed(
        conference_config=conference_config,
        models=models,
        dataloader=dataloader,
        loss_fn=loss_function,
        optimizer_factory=optimizer_factory,
        strategy="multiprocessing",
        conference_name="mlflow_demo_conference",
        num_processes=1
    )
    
    print("Training completed with MLflow tracking!")
    print("Check MLflow UI to see the logged metrics and artifacts.")
    
    return results


def demonstrate_mandate_comparison():
    """Demonstrate comparing different mandates with MLflow."""
    print("\n📊 Mandate Comparison Demo")
    print("=" * 50)
    
    # Create MLflow tracker
    mlflow_tracker = AutocamMLflowTracker(
        experiment_name="mandate_comparison",
        log_artifacts=True,
        log_models=False  # Don't log models for comparison demo
    )
    
    # Create participants
    participants = [
        Participant(
            name=f"model_{i}",
            model_type=ModelType.CNN,
            model_tag="encoder",
            in_channels=3,
            out_channels=128,
            dimension=Dimension.D2,
            config={"layers": 3, "kernel_size": 3}
        )
        for i in range(4)
    ]
    
    # Test different mandates
    mandates = [
        TrainingMandate.ONE_VS_ONE,
        TrainingMandate.ONE_VS_RANDOM_MEAN,
        TrainingMandate.BARYCENTRIC_TARGETS
    ]
    
    all_results = {}
    
    for mandate in mandates:
        print(f"\nTesting mandate: {mandate.value}")
        
        # Create conference with this mandate
        conference_config = create_conference_grid(
            participants=participants,
            num_sessions=1,
            working_groups_per_session=1,
            training_mandate=mandate,
            mixing_strategy="max"
        )
        
        # Create models and trainer
        models = create_models(participants)
        gpu_allocator = GPUAllocator(["gpu:0"])
        trainer = DistributedMandateTrainer(gpu_allocator, mlflow_tracker)
        
        # Create dataloader and optimizer factory
        dataloader = create_dummy_dataloader(num_batches=5)  # Shorter for demo
        optimizer_factory = create_optimizer_factory()
        
        # Train with this mandate
        results = trainer.train_conference_distributed(
            conference_config=conference_config,
            models=models,
            dataloader=dataloader,
            loss_fn=loss_function,
            optimizer_factory=optimizer_factory,
            strategy="multiprocessing",
            conference_name=f"mandate_comparison_{mandate.value}",
            num_processes=1
        )
        
        all_results[mandate.value] = results
    
    print("\nMandate comparison completed!")
    print("Check MLflow UI to compare the different mandates.")
    
    return all_results


def demonstrate_gpu_tracking():
    """Demonstrate GPU tracking with MLflow."""
    print("\n🎯 GPU Tracking Demo")
    print("=" * 50)
    
    # Create MLflow tracker
    mlflow_tracker = AutocamMLflowTracker(
        experiment_name="gpu_tracking",
        log_artifacts=True,
        log_models=False
    )
    
    # Create participants
    participants = [
        Participant(
            name=f"model_{i}",
            model_type=ModelType.CNN,
            model_tag="encoder",
            in_channels=3,
            out_channels=128,
            dimension=Dimension.D2,
            config={"layers": 3, "kernel_size": 3}
        )
        for i in range(6)  # More models to stress GPU
    ]
    
    # Create conference with multiple working groups
    conference_config = create_conference_grid(
        participants=participants,
        num_sessions=2,
        working_groups_per_session=2,
        training_mandate=TrainingMandate.BARYCENTRIC_TARGETS,
        mixing_strategy="max"
    )
    
    # Create models and trainer
    models = create_models(participants)
    gpu_allocator = GPUAllocator(["gpu:0", "gpu:1"])  # Multiple GPUs
    trainer = DistributedMandateTrainer(gpu_allocator, mlflow_tracker)
    
    # Create dataloader and optimizer factory
    dataloader = create_dummy_dataloader(num_batches=8)
    optimizer_factory = create_optimizer_factory()
    
    # Train with GPU tracking
    results = trainer.train_conference_distributed(
        conference_config=conference_config,
        models=models,
        dataloader=dataloader,
        loss_fn=loss_function,
        optimizer_factory=optimizer_factory,
        strategy="multiprocessing",
        conference_name="gpu_tracking_demo",
        num_processes=2
    )
    
    print("GPU tracking completed!")
    print("Check MLflow UI to see GPU usage metrics.")
    
    return results


def demonstrate_mlflow_ui_commands():
    """Show commands to start MLflow UI."""
    print("\n🌐 MLflow UI Commands")
    print("=" * 50)
    print("To view the MLflow UI, run one of these commands:")
    print()
    print("1. Start MLflow UI (default location):")
    print("   mlflow ui")
    print()
    print("2. Start MLflow UI with custom port:")
    print("   mlflow ui --port 5001")
    print()
    print("3. Start MLflow UI with custom host:")
    print("   mlflow ui --host 0.0.0.0 --port 5001")
    print()
    print("4. View specific experiment:")
    print("   mlflow ui --experiment-name autocam_demo")
    print()
    print("The UI will be available at: http://localhost:5000")
    print("You can browse experiments, runs, and compare metrics.")


def main():
    """Run all MLflow demonstrations."""
    print("🎓 MLflow Integration with Autocam")
    print("=" * 60)
    
    # Check if MLflow is available
    try:
        import mlflow
        print("✅ MLflow is available")
    except ImportError:
        print("❌ MLflow not found. Install with: pip install mlflow")
        return
    
    # Run demonstrations
    try:
        # Basic MLflow tracking
        demonstrate_mlflow_tracking()
        
        # Mandate comparison
        demonstrate_mandate_comparison()
        
        # GPU tracking
        demonstrate_gpu_tracking()
        
        # Show UI commands
        demonstrate_mlflow_ui_commands()
        
        print("\n✅ All MLflow demonstrations completed!")
        print("Check the MLflow UI to see the logged experiments and metrics.")
        
    except Exception as e:
        print(f"❌ Error during MLflow demo: {e}")
        print("This might be due to GPU availability or MLflow configuration.")


if __name__ == "__main__":
    main() 