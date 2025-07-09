#!/usr/bin/env python3
"""Example demonstrating distributed training with GPU allocation and mandate protocols."""

import torch
import torch.nn as nn
import torch.optim as optim
from dask.distributed import Client, LocalCluster
import numpy as np

from autocam.models import Participant, ModelType, Dimension, TrainingMandate
from autocam.conference_builder import create_conference_grid
from autocam.distributed_trainer import DistributedMandateTrainer, GPUAllocator


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


def demonstrate_dask_distributed():
    """Demonstrate Dask distributed training."""
    print("🚀 Dask Distributed Training Demo")
    print("=" * 50)
    
    # Create Dask cluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    print(f"Dask cluster: {client}")
    
    # Create participants and conference
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
    
    conference_config = create_conference_grid(
        participants=participants,
        num_sessions=2,
        working_groups_per_session=1,
        training_mandate=TrainingMandate.ONE_VS_ONE,
        mixing_strategy="max"
    )
    
    # Create models and trainer
    models = create_models(participants)
    gpu_allocator = GPUAllocator(["gpu:0", "gpu:1"])  # Assuming 2 GPUs
    trainer = DistributedMandateTrainer(gpu_allocator)
    
    # Create dataloader and optimizer factory
    dataloader = create_dummy_dataloader()
    optimizer_factory = create_optimizer_factory()
    
    # Train conference using Dask
    results = trainer.train_conference_distributed(
        conference_config=conference_config,
        models=models,
        dataloader=dataloader,
        loss_fn=loss_function,
        optimizer_factory=optimizer_factory,
        strategy="dask",
        dask_client=client
    )
    
    print("Training Results:")
    for session_name, session_results in results.items():
        print(f"  Session: {session_name}")
        for wg_name, wg_results in session_results.items():
            print(f"    {wg_name}: {wg_results['gpu_used']}")
            print(f"      Participants: {wg_results['participants']}")
            print(f"      Loss keys: {list(wg_results['results'].keys())}")
    
    client.close()
    cluster.close()


def demonstrate_multiprocessing():
    """Demonstrate multiprocessing training."""
    print("\n🔄 Multiprocessing Training Demo")
    print("=" * 50)
    
    # Create participants and conference
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
    
    conference_config = create_conference_grid(
        participants=participants,
        num_sessions=2,
        working_groups_per_session=1,
        training_mandate=TrainingMandate.BARYCENTRIC_TARGETS,
        mixing_strategy="max"
    )
    
    # Create models and trainer
    models = create_models(participants)
    gpu_allocator = GPUAllocator(["gpu:0"])  # Single GPU for multiprocessing
    trainer = DistributedMandateTrainer(gpu_allocator)
    
    # Create dataloader and optimizer factory
    dataloader = create_dummy_dataloader()
    optimizer_factory = create_optimizer_factory()
    
    # Train conference using multiprocessing
    results = trainer.train_conference_distributed(
        conference_config=conference_config,
        models=models,
        dataloader=dataloader,
        loss_fn=loss_function,
        optimizer_factory=optimizer_factory,
        strategy="multiprocessing",
        num_processes=2
    )
    
    print("Training Results:")
    for session_name, session_results in results.items():
        print(f"  Session: {session_name}")
        for wg_name, wg_results in session_results.items():
            print(f"    {wg_name}: {wg_results['gpu_used']}")
            print(f"      Participants: {wg_results['participants']}")
            print(f"      Loss keys: {list(wg_results['results'].keys())}")


def demonstrate_gpu_allocation_strategies():
    """Demonstrate different GPU allocation strategies."""
    print("\n🎯 GPU Allocation Strategy Demo")
    print("=" * 50)
    
    # Create GPU allocator
    gpu_allocator = GPUAllocator(["gpu:0", "gpu:1", "gpu:2"])
    
    # Test different allocation strategies
    strategies = ["round_robin", "random", "least_used"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        # Reset allocator
        gpu_allocator = GPUAllocator(["gpu:0", "gpu:1", "gpu:2"])
        
        # Allocate GPUs to working groups
        working_groups = ["wg1", "wg2", "wg3", "wg4"]
        allocations = {}
        
        for wg in working_groups:
            gpu_id = gpu_allocator.allocate_gpu(wg, strategy)
            allocations[wg] = gpu_id
            print(f"  {wg} -> {gpu_id}")
        
        # Release some allocations
        gpu_allocator.release_gpu("wg1")
        gpu_allocator.release_gpu("wg3")
        
        # Allocate new ones
        new_gpu1 = gpu_allocator.allocate_gpu("wg5", strategy)
        new_gpu2 = gpu_allocator.allocate_gpu("wg6", strategy)
        print(f"  wg5 -> {new_gpu1}")
        print(f"  wg6 -> {new_gpu2}")


def demonstrate_mandate_protocols_with_gpu():
    """Demonstrate how mandate protocols work with GPU allocation."""
    print("\n🔧 Mandate Protocols with GPU Demo")
    print("=" * 50)
    
    # Create a simple working group
    from autocam.models import WorkingGroup
    
    working_group = WorkingGroup(
        name="test_group",
        description="Test working group",
        participants=["model_0", "model_1", "model_2", "model_3"],
        training_mandate=TrainingMandate.ONE_VS_ONE,
        student_participants=["model_0", "model_1"],
        target_participants=["model_2", "model_3"],
        training_epochs=100
    )
    
    # Create models
    participants = [
        Participant(name=f"model_{i}", model_type=ModelType.CNN, model_tag="encoder",
                   in_channels=3, out_channels=128, dimension=Dimension.D2)
        for i in range(4)
    ]
    models = create_models(participants)
    
    # Create dummy batch
    batch = torch.randn(8, 3, 32, 32)
    
    # Test different mandates
    mandates = [
        TrainingMandate.ONE_VS_ONE,
        TrainingMandate.ONE_VS_RANDOM_MEAN,
        TrainingMandate.BARYCENTRIC_TARGETS
    ]
    
    for mandate in mandates:
        print(f"\nMandate: {mandate.value}")
        
        # Update working group mandate
        working_group.training_mandate = mandate
        
        # Get reference targets
        targets = mandate.get_reference(batch, working_group, models)
        print(f"  Targets: {list(targets.keys())}")
        
        # Get student-target pairs
        pairs = mandate.get_student_target_pairs(working_group)
        print(f"  Pairs: {pairs}")
        
        # Show target shapes
        for student, target in targets.items():
            print(f"    {student}: {target.shape}")


def main():
    """Run all demonstrations."""
    print("🎓 Distributed Training with GPU Allocation and Mandate Protocols")
    print("=" * 70)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run demonstrations
    demonstrate_mandate_protocols_with_gpu()
    demonstrate_gpu_allocation_strategies()
    
    # Only run distributed demos if we have multiple GPUs or can simulate
    if torch.cuda.device_count() >= 1:
        demonstrate_multiprocessing()
        
        # Only run Dask demo if we can create a cluster
        try:
            demonstrate_dask_distributed()
        except Exception as e:
            print(f"⚠️  Dask demo skipped: {e}")
    else:
        print("\n⚠️  Skipping distributed demos (need GPUs)")
    
    print("\n✅ All demonstrations completed!")


if __name__ == "__main__":
    main() 