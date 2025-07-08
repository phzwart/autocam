"""Tests for the Dask + PyTorch orchestrator."""

import pytest
import torch
import torch.nn as nn

from autocam.conference import Conference
from autocam.orchestrator import AutocamDaskOrchestrator
from autocam.orchestrator import GPUResourceManager


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize dummy model."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64), nn.ReLU(), nn.Linear(64, out_channels)
        )

    def forward(self, x):
        """Forward pass."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


def create_test_config():
    """Create a minimal test configuration."""
    return {
        "conference": {
            "name": "Test_Conference",
            "description": "Test conference",
            "participants": [
                {
                    "name": "test_model_1",
                    "model_type": "custom",
                    "model_tag": "encoder",
                    "in_channels": 3,
                    "out_channels": 128,
                    "dimension": "2D",
                    "config": {"layers": 2},
                },
                {
                    "name": "test_model_2",
                    "model_type": "custom",
                    "model_tag": "encoder",
                    "in_channels": 3,
                    "out_channels": 128,
                    "dimension": "2D",
                    "config": {"layers": 2},
                },
            ],
            "parallel_sessions": [
                {
                    "name": "test_session",
                    "description": "Test session",
                    "training_epochs": 10,
                    "working_groups": [
                        {
                            "name": "test_group",
                            "description": "Test group",
                            "participants": ["test_model_1", "test_model_2"],
                            "training_mandate": "one_vs_one",
                            "student_participants": ["test_model_1"],
                            "target_participants": ["test_model_2"],
                            "training_epochs": 10,
                        }
                    ],
                }
            ],
        },
        "metadata": {"version": "1.0.0", "created_by": "autocam"},
    }


def test_gpu_resource_manager():
    """Test GPU resource manager."""
    manager = GPUResourceManager(["gpu:0", "gpu:1", "gpu:2"])

    # Test round-robin allocation
    wg = type("WorkingGroup", (), {"participants": ["model1", "model2"]})()
    allocated = manager.allocate_for_working_group(wg, "round_robin")
    assert len(allocated) == 2
    assert "gpu:0" in allocated
    assert "gpu:1" in allocated


def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    orchestrator = AutocamDaskOrchestrator(n_workers=1, gpus_per_worker=1)
    assert orchestrator.n_workers == 1
    assert orchestrator.gpus_per_worker == 1
    orchestrator.close()


def test_conference_loading():
    """Test loading conference from dict."""
    config_data = create_test_config()
    from autocam.models import ConferenceConfig

    config = ConferenceConfig(**config_data["conference"])

    assert config.name == "Test_Conference"
    assert len(config.participants) == 2
    assert len(config.parallel_sessions) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_orchestrator_with_models():
    """Test orchestrator with actual models (requires CUDA)."""
    config_data = create_test_config()
    conference = Conference.from_dict(config_data)

    # Create model implementations
    model_implementations = {}
    for participant in conference.config.participants:
        model = DummyModel(
            in_channels=participant.in_channels, out_channels=participant.out_channels
        )
        model_implementations[participant.name] = model

    def loss_fn(student_output, target_output):
        return nn.MSELoss()(student_output, target_output)

    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize orchestrator
    orchestrator = AutocamDaskOrchestrator(n_workers=1, gpus_per_worker=1)

    try:
        # This would run the actual conference
        # For now, just test that it initializes correctly
        assert orchestrator.client is not None
        assert orchestrator.cluster is not None
    finally:
        orchestrator.close()


def test_orchestrator_imports():
    """Test that orchestrator can be imported."""
    from autocam.orchestrator import AutocamDaskOrchestrator
    from autocam.orchestrator import GPUResourceManager

    assert AutocamDaskOrchestrator is not None
    assert GPUResourceManager is not None
