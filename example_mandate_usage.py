#!/usr/bin/env python3
"""Example demonstrating the new training mandate protocol system."""

import torch
import numpy as np
from autocam.models import Participant, ModelType, Dimension, TrainingMandate
from autocam.conference_builder import create_conference_grid, demonstrate_mandate_usage


class DummyModel:
    """A dummy model for demonstration purposes."""
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.training_mode = True
    
    def forward(self, x):
        """Forward pass - returns a random tensor of the correct shape."""
        batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
        if isinstance(self.output_shape, tuple) and len(self.output_shape) > 0:
            return torch.randn(batch_size, *self.output_shape)
        else:
            return torch.randn(batch_size)
    
    def train(self):
        self.training_mode = True
    
    def eval(self):
        self.training_mode = False


def create_dummy_participants(num_models: int = 4):
    """Create dummy participants for demonstration."""
    participants = []
    for i in range(num_models):
        participant = Participant(
            name=f"model_{i}",
            model_type=ModelType.CNN,
            model_tag="encoder",
            in_channels=3,
            out_channels=128,
            dimension=Dimension.D2,
            config={"layers": 3, "kernel_size": 3}
        )
        participants.append(participant)
    return participants


def create_dummy_models(participants):
    """Create dummy model implementations."""
    models = {}
    for participant in participants:
        models[participant.name] = DummyModel(
            participant.name,
            (participant.in_channels, 32, 32),
            (participant.out_channels, 16, 16)
        )
    return models


def create_dummy_optimizers(model_names):
    """Create dummy optimizers."""
    optimizers = {}
    for name in model_names:
        # Create a dummy optimizer (in practice, this would be torch.optim.Adam, etc.)
        optimizers[name] = {"optimizer": f"dummy_optimizer_{name}"}
    return optimizers


def main():
    """Demonstrate the training mandate protocol system."""
    print("🚀 Training Mandate Protocol System Demo")
    print("=" * 60)
    
    # Create participants
    participants = create_dummy_participants(4)
    print(f"📋 Created {len(participants)} participants")
    
    # Create conference grid with different mandates
    print("\n🎓 Creating conference with ONE_VS_ONE mandate...")
    conference_config = create_conference_grid(
        participants=participants,
        num_sessions=3,
        working_groups_per_session=1,
        training_mandate=TrainingMandate.ONE_VS_ONE,
        mixing_strategy="max"
    )
    
    # Create dummy models and optimizers
    models = create_dummy_models(participants)
    optimizers = create_dummy_optimizers([p.name for p in participants])
    
    # Create dummy batch
    dummy_batch = torch.randn(16, 3, 32, 32)
    
    # Demonstrate mandate usage
    demonstrate_mandate_usage(conference_config, models, dummy_batch, optimizers)
    
    # Show how to use protocols directly
    print("\n🔧 Direct Protocol Usage Examples:")
    print("-" * 40)
    
    for session in conference_config.parallel_sessions:
        for working_group in session.working_groups:
            print(f"\nWorking Group: {working_group.name}")
            print(f"Mandate: {working_group.training_mandate}")
            
            # Get reference targets
            targets = working_group.training_mandate.get_reference(
                dummy_batch, working_group, models
            )
            print(f"  Targets: {list(targets.keys())}")
            
            # Get student-target pairs
            pairs = working_group.training_mandate.get_student_target_pairs(working_group)
            print(f"  Pairs: {pairs}")
            
            # Get optimizer assignments
            optimizer_assignments = working_group.training_mandate.get_optimizer_assignments(
                working_group, optimizers
            )
            print(f"  Optimizers: {list(optimizer_assignments.keys())}")
    
    # Demonstrate different mandates
    print("\n🔄 Demonstrating Different Mandates:")
    print("-" * 40)
    
    mandates = [
        TrainingMandate.ONE_VS_ONE,
        TrainingMandate.ONE_VS_RANDOM_MEAN,
        TrainingMandate.ONE_VS_FIXED,
        TrainingMandate.RANDOM_PAIRS,
        TrainingMandate.BARYCENTRIC_TARGETS
    ]
    
    for mandate in mandates:
        print(f"\nMandate: {mandate.value}")
        
        # Create a simple working group for this mandate
        simple_wg = type('WorkingGroup', (), {
            'student_participants': ['model_0', 'model_1'],
            'target_participants': ['model_2', 'model_3'],
            'fixed_target': 'model_0',
            'random_mean_count': 2,
            'barycentric_min_models': 2,
            'barycentric_max_models': 3,
            'participants': ['model_0', 'model_1', 'model_2', 'model_3']
        })()
        
        # Get reference targets
        targets = mandate.get_reference(dummy_batch, simple_wg, models)
        print(f"  Targets generated: {list(targets.keys())}")
        
        # Get pairs
        pairs = mandate.get_student_target_pairs(simple_wg)
        print(f"  Pairs: {pairs}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main() 