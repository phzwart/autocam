"""Generated conference configuration for representation learning.

Generated from YAML configuration.
This is a meta-framework for organizing models - actual models will be injected later.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class ModelInterface(ABC):
    """Abstract interface for models that can be injected."""

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get the output shape."""
        pass


class ParticipantRegistry:
    """Registry for managing participant models."""

    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, ModelInterface] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}

    def register_model(self, name: str, model: ModelInterface, spec: Dict[str, Any]):
        """Register a model with its specification."""
        self._models[name] = model
        self._specs[name] = spec

    def get_model(self, name: str) -> Optional[ModelInterface]:
        """Get a model by name."""
        return self._models.get(name)

    def get_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model specification by name."""
        return self._specs.get(name)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_models_by_tag(self, tag: str) -> List[str]:
        """List models by tag."""
        return [
            name for name, spec in self._specs.items() if spec.get("model_tag") == tag
        ]


# Conference configuration
class MinimalConferenceConference:
    """A minimal conference with just one participant and session."""

    def __init__(self):
        """Initialize the conference."""
        self.name = "Minimal_Conference"
        self.registry = ParticipantRegistry()
        self.parallel_sessions = {}
        self._setup_sessions()

    def _setup_sessions(self):
        """Setup parallel sessions and working groups."""
        self.parallel_sessions["Single_Session"] = {
            "name": "Single_Session",
            "description": "A single session with one working group",
            "working_groups": {
                "Single_Group": {
                    "name": "Single_Group",
                    "description": "A single working group",
                    "participant_names": ["simple_model"],
                    "participants": [],  # Will be populated when models are registered
                }
            },
        }

    def register_participant(
        self, name: str, model: ModelInterface, spec: Dict[str, Any]
    ):
        """Register a participant model."""
        self.registry.register_model(name, model, spec)

        # Update working groups with registered models
        for session in self.parallel_sessions.values():
            for wg in session["working_groups"].values():
                if name in wg["participant_names"]:
                    wg["participants"].append(model)

    def get_participant(self, name: str) -> Optional[ModelInterface]:
        """Get a participant model by name."""
        return self.registry.get_model(name)

    def get_participant_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get participant specification by name."""
        return self.registry.get_spec(name)

    def get_working_group(self, session_name: str, group_name: str):
        """Get a working group by session and group name."""
        session = self.parallel_sessions.get(session_name)
        if session:
            return session["working_groups"].get(group_name)
        return None

    def get_session_participants(self, session_name: str) -> List[ModelInterface]:
        """Get all participants in a session."""
        session = self.parallel_sessions.get(session_name)
        if session:
            participants = []
            for group in session["working_groups"].values():
                participants.extend(group["participants"])
            return participants
        return []

    def run_session(self, session_name: str, data: Any):
        """Run all models in a session on the given data."""
        participants = self.get_session_participants(session_name)
        results = {}
        for i, model in enumerate(participants):
            try:
                results[f"model_{i}"] = model.forward(data)
            except Exception as e:
                results[f"model_{i}"] = f"Error: {e}"
        return results

    def get_participants_by_tag(self, tag: str) -> List[str]:
        """Get participant names by tag."""
        return self.registry.list_models_by_tag(tag)

    def validate_registration(self) -> bool:
        """Validate that all expected participants are registered."""
        expected_participants = set()
        for session in self.parallel_sessions.values():
            for wg in session["working_groups"].values():
                expected_participants.update(wg["participant_names"])

        registered_participants = set(self.registry.list_models())
        missing = expected_participants - registered_participants

        if missing:
            print(f"Missing participants: {missing}")
            return False
        return True


# Expected participant specifications
EXPECTED_PARTICIPANTS = {
    "simple_model": {
        "name": "simple_model",
        "model_type": "ModelType.MLP",
        "model_tag": "classifier",
        "in_channels": 10,
        "out_channels": 2,
        "dimension": "Dimension.D2",
        "model_config": {"protected_namespaces": ()},
    },
}

# Main conference instance
conference = MinimalConferenceConference()

if __name__ == "__main__":
    # Example usage
    print("Conference: Minimal_Conference")
    print(f"Expected participants: {list(EXPECTED_PARTICIPANTS.keys())}")
    print(f"Sessions: {list(conference.parallel_sessions.keys())}")

    # Example of how to register models
    print("\nExample model registration:")
    for name, spec in EXPECTED_PARTICIPANTS.items():
        print(
            f"  conference.register_participant({name!r}, your_model_instance, {spec})"
        )

    # Example session structure

    print("\nSession: Single_Session")

    print("  Working Group: Single_Group - A single working group")
    print(f"    Participants: {['simple_model']}")
