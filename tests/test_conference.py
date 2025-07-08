"""Tests for conference configuration."""

import tempfile
from pathlib import Path

from autocam.config import load_conference_config
from autocam.config import save_conference_config
from autocam.config import validate_yaml_schema
from autocam.models import ConferenceConfig


class TestConferenceConfig:
    """Test conference configuration functionality."""

    def __init__(self):
        """Initialize test data."""
        self.sample_config = {
            "conference": {
                "name": "Test_Conference",
                "description": "Test conference",
                "participants": [
                    {
                        "name": "test_model",
                        "model_type": "cnn",
                        "model_tag": "encoder",
                        "in_channels": 3,
                        "out_channels": 64,
                        "dimension": "2D",
                        "config": {"layers": 3},
                    }
                ],
                "parallel_sessions": [
                    {
                        "name": "Test_Session",
                        "description": "Test session",
                        "working_groups": [
                            {
                                "name": "Test_Group",
                                "description": "Test group",
                                "participants": ["test_model"],
                            }
                        ],
                    }
                ],
            },
            "metadata": {"version": "1.0.0", "created_by": "test"},
        }

    def test_load_save_config(self):
        """Test loading and saving configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name

        try:
            # Save config
            config = ConferenceConfig(**self.sample_config)
            save_conference_config(config, temp_file)

            # Load config
            loaded_config = load_conference_config(temp_file)
            assert loaded_config.conference.name == "Test_Conference"
            assert len(loaded_config.conference.participants) == 1
            assert len(loaded_config.conference.parallel_sessions) == 1

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_validate_config(self):
        """Test configuration validation."""
        config = ConferenceConfig(**self.sample_config)
        config.validate_config()  # Should not raise

    def test_validate_yaml_schema(self):
        """Test YAML schema validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name

        try:
            # Save valid config
            config = ConferenceConfig(**self.sample_config)
            save_conference_config(config, temp_file)

            # Validate
            assert validate_yaml_schema(temp_file)

        finally:
            Path(temp_file).unlink(missing_ok=True)
