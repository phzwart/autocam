#!/usr/bin/env python3
"""Minimal integration of dlsia CNN networks with Autocam."""

import numpy as np
import yaml
from autocam.conference import ModelInterface, create_conference


class DlsiaCNNAdapter(ModelInterface):
    """Simple adapter for dlsia CNN networks."""
    
    def __init__(self, dlsia_model, name: str, input_shape: tuple, output_shape: tuple):
        self.dlsia_model = dlsia_model
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, x):
        """Forward pass through dlsia model."""
        if hasattr(self.dlsia_model, 'forward'):
            return self.dlsia_model.forward(x)
        elif hasattr(self.dlsia_model, '__call__'):
            return self.dlsia_model(x)
        else:
            # Fallback for testing
            batch_size = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            return np.random.randn(batch_size, *self.output_shape)
    
    def get_input_shape(self) -> tuple:
        return self.input_shape
    
    def get_output_shape(self) -> tuple:
        return self.output_shape
    
    def train(self):
        if hasattr(self.dlsia_model, 'train'):
            self.dlsia_model.train()
    
    def eval(self):
        if hasattr(self.dlsia_model, 'eval'):
            self.dlsia_model.eval()


# Step 1: Create participants from your dlsia networks
def create_participants(dlsia_networks):
    """Convert your dlsia networks to Autocam participants."""
    participants = {}
    
    # Adjust these parameters based on your actual networks
    input_shape = (3, 32, 32)  # RGB, 32x32
    output_shape = (128,)       # 128-dimensional features
    
    for i, network in enumerate(dlsia_networks):
        name = f"dlsia_cnn_{i}"
        adapter = DlsiaCNNAdapter(network, name, input_shape, output_shape)
        participants[name] = adapter
    
    return participants


# Step 2: Create conference configuration
def create_conference_yaml(participant_names):
    """Create YAML configuration for the conference."""
    config = {
        "conference": {
            "name": "Dlsia_CNN_Conference",
            "description": "Conference for dlsia CNN networks",
            "participants": [
                {
                    "name": name,
                    "model_type": "cnn",
                    "model_tag": "encoder",
                    "in_channels": 3,
                    "out_channels": 128,
                    "dimension": "2D",
                    "config": {"layers": 3, "kernel_size": 3}
                }
                for name in participant_names
            ],
            "parallel_sessions": [
                {
                    "name": "CNN_Comparison_Session",
                    "description": "Compare dlsia CNN architectures",
                    "working_groups": [
                        {
                            "name": "CNN_Group",
                            "description": "CNN comparison group",
                            "participants": participant_names,
                            "training_mandate": "one_vs_one",
                            "student_participants": participant_names[:-1],
                            "target_participants": participant_names[1:],
                            "training_epochs": 10
                        }
                    ]
                }
            ]
        },
        "metadata": {"version": "1.0.0", "created_by": "dlsia"}
    }
    
    return config


# Step 3: Main integration function
def integrate_dlsia_with_autocam(dlsia_networks):
    """Main function to integrate dlsia networks with Autocam."""
    
    print("🔧 Step 1: Creating participants from dlsia networks")
    participants = create_participants(dlsia_networks)
    print(f"✅ Created {len(participants)} participants")
    
    print("\n🔧 Step 2: Creating conference configuration")
    config = create_conference_yaml(list(participants.keys()))
    
    # Save YAML file
    with open("dlsia_conference.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print("✅ Saved dlsia_conference.yaml")
    
    print("\n🔧 Step 3: Loading conference")
    conference = create_conference("dlsia_conference.yaml")
    print(f"✅ Loaded conference: {conference.name}")
    
    print("\n🔧 Step 4: Registering participants")
    for name, participant in participants.items():
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
            conference.register_participant(name, participant, spec)
            print(f"✅ Registered {name}")
    
    print("\n🔧 Step 5: Validating")
    if conference.validate_registration():
        print("✅ All participants registered successfully")
    else:
        print("❌ Some participants missing")
    
    print(f"\n📊 Conference Info:")
    print(f"  Sessions: {conference.list_sessions()}")
    print(f"  Participants: {conference.list_participants()}")
    
    return conference, participants


# Example usage:
if __name__ == "__main__":
    # Replace this with your actual dlsia networks
    class MockDlsiaNetwork:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return f"Mock {self.name} processed: {x.shape}"
    
    # Your dlsia networks would go here
    dlsia_networks = [
        MockDlsiaNetwork("network_1"),
        MockDlsiaNetwork("network_2"),
        MockDlsiaNetwork("network_3")
    ]
    
    # Integrate with Autocam
    conference, participants = integrate_dlsia_with_autocam(dlsia_networks)
    
    # Test the integration
    print("\n🔧 Step 6: Testing integration")
    test_data = np.random.randn(4, 3, 32, 32)
    results = conference.run_session("CNN_Comparison_Session", test_data)
    
    print("📊 Test Results:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result}")
    
    print("\n🎉 Integration complete!") 